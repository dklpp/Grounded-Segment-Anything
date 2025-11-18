import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

print("Loading DINOv2 ViT-L/14 via torch.hub...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval()

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def extract_feature(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat = model(x)
    return feat.cpu().numpy().flatten()


def main(root_folder, output_file="dino_features.npz"):
    features = []
    paths = []

    for subdir, _, files in os.walk(root_folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(subdir, fname)
                rel_path = os.path.relpath(full_path, root_folder)

                print("Extracting:", rel_path)
                vec = extract_feature(full_path)

                features.append(vec)
                paths.append(rel_path)

    np.savez(output_file,
             features=np.array(features),
             paths=np.array(paths))

    print("\nSaved:", output_file)
    print("Total images:", len(paths))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder with many subfolders")
    parser.add_argument("--output", default="dino_features.npz")
    args = parser.parse_args()

    main(args.root, args.output)
