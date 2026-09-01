import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import torch
from tqdm import tqdm

# Vendored official package: NeuroVFM_CVPR26CTFM/neurovfm/neurovfm/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurovfm"))
from neurovfm.pipelines import load_encoder # pyright: ignore[reportMissingImports]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        imgs_files = [
            f for f in imgs_files
            if os.path.exists(os.path.join(args.masks_path, f))
        ]

    n = 0
    for img_file in tqdm(imgs_files, desc="Extracting features"):
        img_id = img_file.replace(".nii.gz", "")
        try:
            batch = preprocessor.load_study(
                os.path.join(args.imgs_path, img_file), modality="ct"
            )
            embs = encoder.embed(batch)
            if embs.ndim != 2 or embs.shape[1] != 768 or embs.shape[0] < 1:
                raise ValueError(
                    f"expected [num_patches, hidden_dim]=[N, 768], got {tuple(embs.shape)}"
                )
            y_hat = embs.float().cpu().numpy()
        except Exception as e:
            print(f"Error extracting {img_file}: {e}")
            continue

        print(f"{img_id}: y_hat.shape={y_hat.shape}")  # [num_patches, 768]
        out = os.path.join(args.dest, f"{img_id}.h5")
        assert not os.path.exists(out), out
        with h5py.File(out, "w") as hf:
            hf.create_dataset("y_hat", data=y_hat)
        n += 1

    print(f"Done. Processed {n}/{len(imgs_files)} images.")
