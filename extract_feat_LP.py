import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Vendored official package: NeuroVFM_CVPR26CTFM/neurovfm/neurovfm/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurovfm"))
from neurovfm.pipelines import load_encoder # pyright: ignore[reportMissingImports]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_pool(embs: torch.Tensor) -> np.ndarray:
    if embs.ndim != 2 or embs.shape[0] == 0:
        raise ValueError(f"expected non-empty [N, D] embeddings, got {tuple(embs.shape)}")
    y_hat = embs.mean(dim=0).float().cpu().numpy()
    if y_hat.shape != (768,) or not np.isfinite(y_hat).all():
        raise ValueError(f"bad y_hat shape={y_hat.shape}")
    return y_hat


def extract_one(encoder, preprocessor, img_path: str) -> np.ndarray:
    batch = preprocessor.load_study(img_path, modality="ct")
    embs = encoder.embed(batch)
    return average_pool(embs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", default="/workspace/inputs")
    ap.add_argument("-o", "--output", "--dest", dest="dest", default="/workspace/outputs")
    ap.add_argument("--masks_path", default=None)
    ap.add_argument("--checkpoint", default="./checkpoints/neurovfm-encoder")
    args = ap.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    encoder, preprocessor = load_encoder(args.checkpoint, device=str(device))
    os.makedirs(args.dest, exist_ok=True)

    imgs_files = sorted(f for f in os.listdir(args.imgs_path) if f.endswith(".nii.gz"))
    if args.masks_path:
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(args.masks_path, f))]

    n = 0
    for img_file in tqdm(imgs_files, desc="Extracting features"):
        img_id = img_file.replace(".nii.gz", "")
        try:
            y_hat = extract_one(encoder, preprocessor, os.path.join(args.imgs_path, img_file))
        except Exception as e:
            print(f"Error extracting {img_file}: {e}")
            continue

        out = os.path.join(args.dest, f"{img_id}.h5")
        assert not os.path.exists(out), out
        with h5py.File(out, "w") as hf:
            hf.create_dataset("y_hat", data=y_hat)
        n += 1

    print(f"Done. Processed {n}/{len(imgs_files)} images.")
