# NeuroVFM_CVPR26CTFM

CTFM linear-probing feature extraction with a frozen [NeuroVFM](https://github.com/MLNeurosurg/neurovfm) vision encoder.

The official NeuroVFM Python package is vendored under `neurovfm/` (commit `9240021d`) so `from neurovfm.pipelines import load_encoder` works offline. Do not rely on installing the package from GitHub at evaluation time.

| Path | Role |
|---|---|
| `neurovfm/` | Official [MLNeurosurg/neurovfm](https://github.com/MLNeurosurg/neurovfm) source (`pip install -e ./neurovfm`) |
| `extract_feat_LP.py` | Read `/workspace/inputs/*.nii.gz`, average-pool encoder tokens, write `{id}.h5` (`y_hat`) |
| `extract_feat_LP.sh` | Docker entrypoint using `INPUT_DIR` / `OUTPUT_DIR` / optional `MASKS_DIR` |

`extract_feat_LP.py` handles both evaluation passes with one code path. Without
`--masks_path` it embeds the whole image; with `--masks_path` it resamples the ROI
mask onto the preprocessed image grid and crops to the mask bounding box before
tokenization. Both passes reuse NeuroVFM's own `load_image` / `prepare_for_inference`
/ `tokenize_volume`, so preprocessing matches the official pipeline exactly.
| `Dockerfile` | Install the vendored package, bake encoder weights, run offline |

Do not push `checkpoints/` or CT-CLIP. Weights belong only in the `docker build` artifact `neurovfm_lp.tar.gz`.
