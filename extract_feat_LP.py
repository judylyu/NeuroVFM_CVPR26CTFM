import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

# Vendored official package: NeuroVFM_CVPR26CTFM/neurovfm/neurovfm/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurovfm"))
from neurovfm.data.io import load_image  # pyright: ignore[reportMissingImports]
from neurovfm.data.preprocess import (  # pyright: ignore[reportMissingImports]
    prepare_for_inference,
    tokenize_volume,
    transpose_to_dhw,
)
from neurovfm.pipelines import load_encoder  # pyright: ignore[reportMissingImports]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_pool(embs: torch.Tensor) -> np.ndarray:
    if embs.ndim != 2 or embs.shape[0] == 0:
        raise ValueError(f"expected non-empty [N, D] embeddings, got {tuple(embs.shape)}")
    y_hat = embs.mean(dim=0).float().cpu().numpy()
    if y_hat.shape != (768,) or not np.isfinite(y_hat).all():
        raise ValueError(f"bad y_hat shape={y_hat.shape}")
    return y_hat


def foreground_box(mask, patch_size, margin):
    # Foreground crop: bounding box of the ROI mask, grown by `margin` patches on
    # each side and snapped outward to whole patches, so the cropped volume stays
    # divisible by patch_size (tokenize_volume asserts that) and keeps at least
    # one patch per axis (prepare_for_inference requires D>=4, H>=16, W>=16).
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("mask has no foreground voxels")
    lo, hi = coords.min(0), coords.max(0) + 1
    box = []
    for dim, patch in enumerate(patch_size):
        start = max(int(lo[dim]) - margin * patch, 0) // patch * patch
        stop = min(int(hi[dim]) + margin * patch, mask.shape[dim])
        stop = min(-(-stop // patch) * patch, mask.shape[dim])
        box.append(slice(start, max(stop, start + patch)))
    return tuple(box)


def load_study(preprocessor, img_path, mask_path, margin, modality="ct"):
    """StudyPreprocessor.load_study for a single volume, with an optional ROI crop."""
    img_sitk = load_image(img_path, preprocess=True)
    if img_sitk is None:
        raise ValueError(f"failed to load {img_path}")

    result = prepare_for_inference(img_sitk, mode=modality)
    if result is None:
        raise ValueError(f"volume too small after preprocessing: {img_path}")
    img_arrs, background_mask, view = result

    if mask_path is not None:
        # Foreground crop: resample the ROI mask onto the preprocessed image grid
        # (nearest neighbour, so labels stay binary) and transpose it the same way
        # prepare_for_inference transposed the image, then crop every window and
        # the background mask to the mask bounding box before tokenization.
        mask_sitk = load_image(mask_path, preprocess=False)
        if mask_sitk is None:
            raise ValueError(f"failed to load {mask_path}")
        mask_sitk = sitk.Resample(
            mask_sitk, img_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0
        )
        mask_arr, _ = transpose_to_dhw(sitk.GetArrayFromImage(mask_sitk), 2 - view)
        box = foreground_box(mask_arr > 0, preprocessor.patch_size, margin)
        img_arrs = [img_arr[box] for img_arr in img_arrs]
        background_mask = background_mask[box]

    all_tokens, all_coords, series_lengths = [], [], []
    for img_arr in img_arrs:
        tokens, coords, _ = tokenize_volume(
            img_arr,
            background_mask,
            patch_size=preprocessor.patch_size,
            remove_background=preprocessor.remove_background,
        )
        all_tokens.append(torch.from_numpy(tokens).float())
        all_coords.append(torch.from_numpy(coords).long())
        series_lengths.append(len(tokens))

    series_cu_seqlens = torch.zeros(len(series_lengths) + 1, dtype=torch.int32)
    series_cu_seqlens[1:] = torch.tensor(series_lengths, dtype=torch.int32).cumsum(0)

    name = os.path.basename(img_path)
    if modality == "ct":
        windows = ("BrainWindow", "BloodWindow", "BoneWindow")
        paths = [f"{name}_{window}" for window in windows]
    else:
        paths = [name]

    return {
        "img": torch.cat(all_tokens, dim=0),
        "coords": torch.cat(all_coords, dim=0),
        "series_masks_indices": torch.tensor([]),
        "series_cu_seqlens": series_cu_seqlens,
        "series_max_len": max(series_lengths),
        "study_cu_seqlens": torch.tensor([0, series_cu_seqlens[-1]], dtype=torch.int32),
        "study_max_len": len(series_lengths),
        "mode": [modality] * len(series_lengths),
        "path": paths,
        "size": [img_arr.shape for img_arr in img_arrs],
    }


def extract_one(encoder, preprocessor, img_path: str, mask_path, margin: int) -> np.ndarray:
    batch = load_study(preprocessor, img_path, mask_path, margin)
    embs = encoder.embed(batch)
    return average_pool(embs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", default="/workspace/inputs")
    ap.add_argument("-o", "--output", "--dest", dest="dest", default="/workspace/outputs")
    ap.add_argument("--masks_path", default=None, help="ROI masks; omit for whole-image features")
    ap.add_argument("--checkpoint", default="./checkpoints/neurovfm-encoder")
    ap.add_argument("--margin", type=int, default=1, help="patches of context around the ROI")
    args = ap.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if args.masks_path is not None and not os.path.isdir(args.masks_path):
        raise FileNotFoundError(args.masks_path)

    encoder, preprocessor = load_encoder(args.checkpoint, device=str(device))
    os.makedirs(args.dest, exist_ok=True)

    imgs_files = sorted(f for f in os.listdir(args.imgs_path) if f.endswith(".nii.gz"))

    n = 0
    for img_file in tqdm(imgs_files, desc="Extracting features"):
        mask_full = None
        if args.masks_path is not None:
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
