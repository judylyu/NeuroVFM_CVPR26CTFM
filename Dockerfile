FROM python:3.10.14-bookworm AS py
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

COPY --from=py /usr/local /usr/local

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ninja-build build-essential tzdata \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=user:user . /opt/app/
WORKDIR /opt/app/

RUN pip install --user torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --user torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

RUN pip install --user -e ./neurovfm

RUN pip install --user h5py

ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=8.9
RUN pip install --user flash-attn==2.6.3 --no-build-isolation

# The flash-attn wheel does not build fused_dense_lib, so FusedDense/FusedMLP
# import as None and neurovfm's ViT cannot be constructed. Build the extension.
RUN pip install --user --no-build-isolation \
    "git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3#subdirectory=csrc/fused_dense_lib"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV CHECKPOINT=/opt/app/checkpoints/neurovfm-encoder

WORKDIR /opt/app
ENTRYPOINT []
