export DETECTRON2_DATASETS=/n/pfister_lab/Users/ztchen/
python tools/train_net.py \
  --config-file configs/COCO-KeypointsMask/keypoint_rcnn_R_50_FPN_1x_pretrained.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005