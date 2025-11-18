CUDA_VISIBLE_DEVICES=0 python grounded_dino_crops.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --folder office_dataset \
  --output_dir output_gdino/keyboard_crops \
  --text_prompt "keyboard" \
  --device cuda \
  --box_threshold 0.35 \
  --text_threshold 0.25
