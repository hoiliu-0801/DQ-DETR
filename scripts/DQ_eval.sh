coco_path=$1
checkpoint=$2
python main_aitod.py \
  --output_dir logs/DQ_eval \
	-c config/DQ_5scale.py --coco_path "$coco_path" \
	--eval --resume "$checkpoint" "${@:3}" \
	--options dn_scalar=100 embed_init_tgt=False \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0