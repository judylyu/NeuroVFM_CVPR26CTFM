# NeuroVFM_CVPR26CTFM

Mentor 要求 GitHub 里只放改好的三个文件：

| 文件 | 作用 |
|---|---|
| `extract_feat_LP.py` | 读 `/workspace/inputs/*.nii.gz`，NeuroVFM vision encoder + average pooling，写出 `{id}.h5`（`y_hat`） |
| `extract_feat_LP.sh` | Docker 入口：`INPUT_DIR` / `OUTPUT_DIR` / 可选 `MASKS_DIR` |
| `Dockerfile` | `pip install` NeuroVFM、bake encoder 权重；评测时离线跑 |

不要把 CT-CLIP 源码树、BERT、`checkpoints/` 推进这个 repo。权重只进 `docker build` 产物 `neurovfm_lp.tar.gz`。
