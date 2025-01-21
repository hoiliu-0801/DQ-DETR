coco_path=$1
checkpoint=$2
python -m torch.distributed.launch main_aitod.py \
  --output_dir logs/DQDETR_ver1 \
  -c config/DQ_5scale.py --coco_path "$coco_path" \
  --pretrain_model_path "$checkpoint" "${@:3}" \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0