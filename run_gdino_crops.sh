CUDA_VISIBLE_DEVICES=2 python grounded_dino_crops.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --folder office_dataset/0cubicle \
  --output_dir output_gdino_0cubicle/cabel \
  --text_prompt "cabel" \
  --device cuda \
  --box_threshold 0.35 \
  --text_threshold 0.25
