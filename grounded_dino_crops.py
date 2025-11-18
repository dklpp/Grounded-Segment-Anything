import argparse
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_image(image_path):
    """Loads and preprocesses the image for GroundingDINO."""
    img_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor, _ = transform(img_pil, None)
    return img_pil, img_tensor


def load_model(config_path, checkpoint_path, bert_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    args.bert_base_uncased_path = bert_path

    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    model.to(device)

    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]      # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]                  # (nq, 4)

    # filter boxes by confidence
    mask = logits.max(dim=1)[0] > box_threshold
    logits = logits[mask]
    boxes = boxes[mask]

    if boxes.size(0) == 0:
        return [], []

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    phrases = []
    for logit, box in zip(logits, boxes):
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        phrases.append(phrase)

    return boxes, phrases


def crop_and_save(image_pil, boxes, out_folder, base_name):
    """Crops using bounding boxes (keeps background) and saves all crops in one folder."""
    W, H = image_pil.size
    boxes_px = []

    # Convert normalized xywh → absolute xyxy
    for b in boxes:
        cx, cy, w, h = b
        w *= W
        h *= H
        cx *= W
        cy *= H

        x0 = int(cx - w / 2)
        y0 = int(cy - h / 2)
        x1 = int(cx + w / 2)
        y1 = int(cy + h / 2)

        boxes_px.append([x0, y0, x1, y1])

    saved = 0

    for i, (x0, y0, x1, y1) in enumerate(boxes_px):
        crop = image_pil.crop((x0, y0, x1, y1))
        crop_path = os.path.join(out_folder, f"{base_name}_crop_{i}.jpg")
        crop.save(crop_path)
        saved += 1

    return saved


def process_folder(model, folder_path, out_folder, prompt, box_th, text_th, device):
    os.makedirs(out_folder, exist_ok=True)

    images = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith(("jpg", "png", "jpeg"))])

    for idx, img_name in enumerate(images):
        img_path = os.path.join(folder_path, img_name)
        print(f"[{idx+1}/{len(images)}] Processing {img_name}")

        image_pil, image_tensor = load_image(img_path)
        boxes, _ = get_grounding_output(model, image_tensor, prompt, box_th, text_th, device)

        if len(boxes) == 0:
            print(" → No detections")
            continue

        base_name = os.path.splitext(img_name)[0]
        saved = crop_and_save(image_pil, boxes, out_folder, base_name)

        print(f" → Saved {saved} crops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grounded_checkpoint", required=True)
    parser.add_argument("--folder", required=True, help="Input folder with images")
    parser.add_argument("--output_dir", required=True, help="Folder to save ALL crops")
    parser.add_argument("--text_prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bert_base_uncased_path", default=None)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    args = parser.parse_args()

    model = load_model(args.config,
                       args.grounded_checkpoint,
                       args.bert_base_uncased_path,
                       args.device)

    process_folder(
        model=model,
        folder_path=args.folder,
        out_folder=args.output_dir,
        prompt=args.text_prompt,
        box_th=args.box_threshold,
        text_th=args.text_threshold,
        device=args.device
    )
