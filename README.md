# NeuroVFM_CVPR26CTFM

CTFM linear-probing feature extraction with a frozen [NeuroVFM](https://github.com/MLNeurosurg/neurovfm) vision encoder.

The official NeuroVFM Python package is vendored under `neurovfm/` (commit `9240021d`) so `from neurovfm.pipelines import load_encoder` works offline. Do not rely on installing the package from GitHub at evaluation time.

| Path | Role |
|---|---|
| `neurovfm/` | Official [MLNeurosurg/neurovfm](https://github.com/MLNeurosurg/neurovfm) source (`pip install -e ./neurovfm`) |
| `extract_feat_LP.py` | Read `/workspace/inputs/*.nii.gz`, mean-pool each CT window separately, concatenate brain/blood/bone features, and write a 2304-d `y_hat` |
| `extract_feat_LP.sh` | Docker entrypoint using `INPUT_DIR`, `OUTPUT_DIR`, optional `MASKS_DIR`, and `CHECKPOINT` |
| `Dockerfile` | Install the vendored package, bake encoder weights, and run offline |

`extract_feat_LP.py` uses NeuroVFM's official whole-volume preprocessing:
RPI orientation, 1×1×4 mm resampling, and a small center crop to patch-size
multiples. ROI masks are not used for cropping. When `MASKS_DIR` is provided,
it is used only to select inputs with matching mask files.

For each CT volume, brain, blood, and bone window tokens are mean-pooled
separately and concatenated in that order, producing a 2304-dimensional
`y_hat`.

Do not push `checkpoints/` or CT-CLIP. Weights belong only in the `docker build` artifact `neurovfm_lp.tar.gz`.
