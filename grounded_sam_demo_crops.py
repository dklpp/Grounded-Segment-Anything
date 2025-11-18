import argparse
import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add local repo paths
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import SamPredictor, sam_model_registry, sam_hq_model_registry


# ---------------------------
# Helpers
# ---------------------------
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_path
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    for logit in logits_filt:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(phrase)

    return boxes_filt, pred_phrases


def save_cropped_object(image_rgb, mask, box, save_dir, index):
    """Crop using SAM mask and save."""
    mask_np = mask.cpu().numpy()[0].astype(np.uint8)
    masked = image_rgb.copy()
    masked[mask_np == 0] = 0  # Background removal

    x1, y1, x2, y2 = map(int, box)
    crop = masked[y1:y2, x1:x2]

    out_path = os.path.join(save_dir, f"object_{index:04d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(exts)]
    return sorted(files)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-SAM Cropping Demo")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_version", type=str, default="vit_h")

    parser.add_argument("--input_image", type=str, help="single image")
    parser.add_argument("--input_dir", type=str, help="folder with images")
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bert_base_uncased_path", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load GroundingDINO
    model = load_model(
        args.config,
        args.grounded_checkpoint,
        args.bert_base_uncased_path,
        device=args.device,
    )

    # Load SAM
    sam_model = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint)
    sam_model.to(device=args.device)
    predictor = SamPredictor(sam_model)

    # Choose input mode
    if args.input_dir:
        image_paths = list_images(args.input_dir)
    else:
        image_paths = [args.input_image]

    print(f"Processing {len(image_paths)} images...\n")

    # ---------------------------
    # Process each image
    # ---------------------------
    for img_idx, image_path in enumerate(image_paths):
        print(f"=== [{img_idx+1}/{len(image_paths)}] Processing: {image_path}")

        # create a folder for this image crop
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # load image
        image_pil, image_tensor = load_image(image_path)
        image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # GroundingDINO detection
        boxes_filt, _ = get_grounding_output(
            model,
            image_tensor,
            args.text_prompt,
            args.box_threshold,
            args.text_threshold,
            device=args.device,
        )

        # SAM segmentation
        predictor.set_image(image_rgb)

        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image_rgb.shape[:2]
        ).to(args.device)

        if boxes_filt.shape[0] == 0:
            print("No objects detected for this image — skipping.")
            continue


        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Save cropped objects
        crop_dir = os.path.join(image_output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        for idx, (mask, box) in enumerate(zip(masks, boxes_filt)):
            save_cropped_object(image_rgb, mask, box.numpy(), crop_dir, idx)

        print(f"Saved {len(masks)} crops → {crop_dir}")
