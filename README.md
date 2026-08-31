# NeuroVFM_CVPR26CTFM

NeuroVFM vision-encoder feature extraction for [CVPR 2026 CTFM](https://www.codabench.org/competitions/12650/) linear probing.

Deliverables: this GitHub repo, `Dockerfile`, and `neurovfm_lp.tar.gz`.

- Vision encoder only (no text encoder / LLM)
- One 3D scan per study
- Average pooling over frozen NeuroVFM tokens → `y_hat` in `{id}.h5`
