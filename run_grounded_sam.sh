# CUDA_VISIBLE_DEVICES=0 python grounded_sam_demo_crops.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --grounded_checkpoint groundingdino_swint_ogc.pth \
#   --sam_checkpoint sam_vit_h_4b8939.pth \
#   --input_image assets/demo1.jpg \
#   --output_dir "outputs" \
#   --box_threshold 0.3 \
#   --text_threshold 0.25 \
#   --text_prompt "bear" \
#   --device "cuda"

CUDA_VISIBLE_DEVICES=0 python grounded_sam_demo_crops.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_dir office_dataset/ \
  --text_prompt "computer" \
  --output_dir out/computers \
  --device cuda \
  --box_threshold 0.5 \
  --text_threshold 0.5
