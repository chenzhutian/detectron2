export DETECTRON2_DATASETS=/n/pfister_lab/Users/ztchen/
cd tools/ 
python train_net.py \
  --config-file ../configs/COCO-KeypointsMask/keypoint_rcnn_R_50_FPN_3x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025