import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.modeling.backbone import RegNet
from detectron2.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.data.coco_keypoint import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.optim import SGD as optimizer
from ..common.train import train

# train from scratch
train.init_checkpoint = ""
train.amp.enabled = True
train.ddp.fp16_compression = True
model.backbone.bottom_up.freeze_at = 0

# SyncBN
# fmt: off
model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = "SyncBN"

# model.backbone.bottom_up.norm = "BN" 
# model.backbone.norm = "BN"

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.box_head.conv_norm = \
    model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
                                                                       stats_mode="N")
# fmt: on

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

model.roi_heads.update(
    num_classes=1,
    keypoint_in_features=["p2", "p3", "p4", "p5"],
    keypoint_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_keypoints=13,
        conv_dims=[512] * 8,
        loss_normalizer="visible",
    ),
)

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# Detectron1 uses 2000 proposals per-batch, but this option is per-image in detectron2.
# 1000 proposals per-image is found to hurt box AP.
# Therefore we increase it to 1500 per-image.
model.proposal_generator.post_nms_topk = (1500, 1000)

# Keypoint AP degrades (though box AP improves) when using plain L1 loss
model.roi_heads.box_predictor.smooth_l1_beta = 0.5

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
    L(T.RandomFlip)(horizontal=True),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True
dataloader.train.mapper.use_instance_mask = True

# larger batch-size.
dataloader.train.total_batch_size = 64

dataloader.evaluator.kpt_oks_sigmas = [d / 10.0 for d in [.26, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]]

# Equivalent to 100 epochs.
# 100 ep = 88436 iters * 64 images/iter / 56599 images/ep
train.max_iter = int(200 * 56599 / dataloader.train.total_batch_size)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter * 0.88), int(train.max_iter * 0.96)],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer.lr = 0.1 / (64 / dataloader.train.total_batch_size)
optimizer.weight_decay = 4e-5

# Config source:
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x.py  # noqa
model.backbone.bottom_up = L(RegNet)(
    stem_class=SimpleStem,
    stem_width=32,
    block_class=ResBottleneckBlock,
    depth=23,
    w_a=38.65,
    w_0=96,
    w_m=2.43,
    group_width=40,
    norm="SyncBN",
    out_features=["s1", "s2", "s3", "s4"],
)
model.pixel_std = [57.375, 57.120, 58.395]

# RegNets benefit from enabling cudnn benchmark mode
train.cudnn_benchmark = True
train.output_dir = '/n/pfister_lab/Users/ztchen/coco_keypoint_new_baseline_100ep_3'
