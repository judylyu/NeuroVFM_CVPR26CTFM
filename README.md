# NeuroVFM_CVPR26CTFM

This repo should contain only these three files:

| File | Role |
|---|---|
| `extract_feat_LP.py` | Read `/workspace/inputs/*.nii.gz`, run the NeuroVFM vision encoder with average pooling, write `{id}.h5` (`y_hat`) |
| `extract_feat_LP.sh` | Docker entrypoint using `INPUT_DIR` / `OUTPUT_DIR` / optional `MASKS_DIR` |
| `Dockerfile` | Install NeuroVFM, bake the encoder weights, run offline at evaluation time |

Do not push the CT-CLIP source tree, BERT, or `checkpoints/` to this repo. Weights belong only in the `docker build` artifact `neurovfm_lp.tar.gz`.
