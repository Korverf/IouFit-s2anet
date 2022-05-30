CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/hrsc2016/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016.py

CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/hrsc2016/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016.py \
    work_dirs/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016_1_rerun/epoch_72.pth \
    --out work_dirs/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016_1_rerun/results.pkl \
    --data hrsc2016
