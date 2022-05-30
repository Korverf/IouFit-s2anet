#python tools/test.py configs/dota/retinanet_obb_r50_fpn_1x_dota_gwd.py \
#    work_dirs/retinanet_obb_r50_fpn_1x_dota_gwd/epoch_12.pth \
#    --out work_dirs/retinanet_obb_r50_fpn_1x_dota_gwd/results.pkl

#python tools/test.py configs/hrsc2016/s2anet_r50_fpn_3x_hrsc2016.py \
#    work_dirs/s2anet_r50_fpn_3x_hrsc2016/epoch_36.pth \
#    --out work_dirs/s2anet_r50_fpn_3x_hrsc2016/results.pkl \
#    --data hrsc2016

python tools/train.py configs/dota/retinanet_obb_r50_fpn_1x_dota_piou.py