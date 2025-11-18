import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


# -------------------------
# Load DINOv2 Model
# -------------------------
def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval()
    model.cuda()
    return model


# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------
# Extract Single Image Feature
# -------------------------
def extract_feature(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        feat = model(x)

    feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu().numpy()


# -------------------------
# Query Logic with CATEGORY FILTERING
# -------------------------
def query_topk(query_img, feature_path, k=5):
    print(f"Loading feature DB: {feature_path}")
    data = np.load(feature_path, allow_pickle=True)

    features = data["features"]
    paths = data["paths"]

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Determine category folder based on query path
    query_category = os.path.basename(os.path.dirname(query_img))
    print(f"Detected query category = '{query_category}'")

    # Filter DB to only that category
    filtered_indices = [i for i, p in enumerate(paths) if query_category in p]

    if len(filtered_indices) == 0:
        raise ValueError(f"No images found in category '{query_category}'")

    print(f"Found {len(filtered_indices)} images in same category.")

    filtered_features = features[filtered_indices]
    filtered_paths = np.array(paths)[filtered_indices]

    filtered_features = filtered_features / np.linalg.norm(filtered_features, axis=1, keepdims=True)

    # Load DINOv2 model
    print("Loading DINOv2 model...")
    model = load_dinov2()

    # Extract feature of the query
    print(f"Extracting features for: {query_img}")
    q_feat = extract_feature(model, query_img)

    # Cosine similarity
    sims = filtered_features @ q_feat
    topk_idx = sims.argsort()[::-1][:k]

    print("\nTop-{} similar results within category '{}':".format(k, query_category))

    results = []
    for rank, idx in enumerate(topk_idx, start=1):
        print(f"{rank}. {filtered_paths[idx]} (cos: {sims[idx]:.4f})")
        results.append((filtered_paths[idx], float(sims[idx])))

    return results


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    query_topk(args.query, args.features, args.topk)
