import argparse
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import torch
from nibabel.loadsave import load as nib_load
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurovfm"))
from neurovfm.pipelines import load_encoder  # pyright: ignore[reportMissingImports]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_pool(embs: torch.Tensor) -> np.ndarray:
    if embs.ndim != 2 or embs.shape[0] == 0:
        raise ValueError(f"expected non-empty [N, D] embeddings, got {tuple(embs.shape)}")
    y_hat = embs.mean(dim=0).float().cpu().numpy()
    if y_hat.shape != (768,) or not np.isfinite(y_hat).all():
        raise ValueError(f"bad y_hat shape={y_hat.shape}")
    return y_hat


def crop_around_center(arr, center, size, pad_value=-1000.0):
    zc, yc, xc = [int(c) for c in center]
    dz, dy, dx = [max(int(s), 1) for s in size]
    z0, y0, x0 = zc - dz // 2, yc - dy // 2, xc - dx // 2
    out = np.full((dz, dy, dx), pad_value, dtype=arr.dtype)
    zs, ys, xs = max(z0, 0), max(y0, 0), max(x0, 0)
    ze = min(z0 + dz, arr.shape[0])
    ye = min(y0 + dy, arr.shape[1])
    xe = min(x0 + dx, arr.shape[2])
    out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = arr[zs:ze, ys:ye, xs:xe]
    return out, (z0, y0, x0)


def extract_one(encoder, preprocessor, img_path: str, mask_path: str, margin: int) -> np.ndarray:
    img = nib_load(img_path)
    mask_nii = nib_load(mask_path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    mask = np.asanyarray(mask_nii.dataobj)  # pyright: ignore[reportAttributeAccessIssue]
    if mask.shape != data.shape:
        raise ValueError(f"image/mask shape mismatch: {data.shape} vs {mask.shape}")

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("mask has no foreground voxels")
    center = coords.mean(axis=0).astype(int)
    # bbox + margin, at least 16^3 so NeuroVFM's D>=4 / H>=16 / W>=16 check can pass
    size = np.maximum(coords.max(0) - coords.min(0) + 1 + 2 * margin, 16).astype(int)
    cropped, origin = crop_around_center(data, center, size)

    aff = np.array(img.affine)  # pyright: ignore[reportAttributeAccessIssue]
    aff[:3, 3] = (aff @ np.array([origin[0], origin[1], origin[2], 1.0], dtype=float))[:3]
    cropped_nii = Nifti1Image(cropped, aff)

    fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        nib_save(cropped_nii, tmp_path)
        batch = preprocessor.load_study(tmp_path, modality="ct")
        embs = encoder.embed(batch)
        return average_pool(embs)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", default="/workspace/inputs")
    ap.add_argument("-o", "--output", "--dest", dest="dest", default="/workspace/outputs")
    ap.add_argument("--masks_path", required=True)
    ap.add_argument("--checkpoint", default="./checkpoints/neurovfm-encoder")
    ap.add_argument("--margin", type=int, default=32)
    args = ap.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not os.path.isdir(args.masks_path):
        raise FileNotFoundError(args.masks_path)

    encoder, preprocessor = load_encoder(args.checkpoint, device=str(device))
    os.makedirs(args.dest, exist_ok=True)

    imgs_files = sorted(f for f in os.listdir(args.imgs_path) if f.endswith(".nii.gz"))

    n = 0
    for img_file in tqdm(imgs_files, desc="Extracting ROI features"):
        mask_full = os.path.join(args.masks_path, img_file)
        if not os.path.exists(mask_full):
            print(f"Skipping {img_file}: no matching mask")
            continue
        img_id = img_file.replace(".nii.gz", "")
        try:
            y_hat = extract_one(
                encoder,
                preprocessor,
                os.path.join(args.imgs_path, img_file),
                mask_full,
                args.margin,
            )
        except Exception as e:
            print(f"Error extracting {img_file}: {e}")
            continue

        out = os.path.join(args.dest, f"{img_id}.h5")
        assert not os.path.exists(out), out
        with h5py.File(out, "w") as hf:
            hf.create_dataset("y_hat", data=y_hat)
        n += 1

    print(f"Done. Processed {n}/{len(imgs_files)} images.")
