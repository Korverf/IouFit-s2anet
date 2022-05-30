python train.py configs/ablation_study/nogen_Nor_IOUfit_retinanet_r50_6x_hrsc2016.py

python test.py configs/ablation_study/nogen_Nor_IOUfit_retinanet_r50_6x_hrsc2016.py \
    work_dirs/nogen_Nor_IOUfit_retinanet_r50_6x_hrsc2016_1/epoch_72.pth \
    --out work_dirs/nogen_Nor_IOUfit_retinanet_r50_6x_hrsc2016_1/results.pkl \
    --data hrsc2016
#python train.py configs/hrsc2016/IOUfit_fc3_3_retinanet_obb_r50_fpn_6x_hrsc2016.py
#
#python test.py configs/hrsc2016/IOUfit_fc3_3_retinanet_obb_r50_fpn_6x_hrsc2016.py \
#    work_dirs/IOUfit_fc3_3_retinanet_obb_r50_fpn_6x_hrsc2016_1/epoch_72.pth \
#    --out work_dirs/IOU#fit_fc3_3_retinanet_obb_r50_fpn_6x_hrsc2016_1/results.pkl \
#    --data hrsc2016

#python tools/train.py configs/dota/s2anet_r50_fpn_1x_dota_gwd.py
#
#python tools/test.py configs/dota/s2anet_r50_fpn_1x_dota_gwd.py \
#    work_dirs/s2anet_r50_fpn_1x_dota_gwd_6/epoch_12.pth \
#    --out work_dirs/s2anet_r50_fpn_1x_dota_gwd_6/results.pkl
