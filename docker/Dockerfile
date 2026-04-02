# syntax=docker/dockerfile:1.7-labs

# Canonical Raramuri inference OCI contract.
#
# This Dockerfile is the single authoritative environment definition for Raramuri.
#
# Contract owned here:
# - Base image family: local-first CUDA userspace on arm64, buildable on GB10 and
#   authoritative for Raramuri inference runtime
# - Python toolchain: CPython 3.11 from the image's system interpreter to match the
#   currently available CUDA-enabled aarch64 PyTorch wheel set and tribev2 floor
# - GPU expectation: Hopper or Blackwell class NVIDIA GPU with CUDA 12.8+ support; no emulation or CPU-only
#   x64 cloud variants are downstream derivatives, not the authority
# - Local acceptance target: DGX Spark / GB10 local OCI runtime consumes this image
#   directly with bluey.mp4 as the designated acceptance input; no host-side package
#   install or alternate environment definition is allowed
# - Cache roots: HF_HOME=/models/hf and TRIBE_CACHE=/models/tribe
# - Secret boundary: tokens are runtime-only env injection; no secrets copied into
#   layers or source-controlled config
# - Entrypoint expectation: `python /app/infer.py <video-or-url> --output <path>`
#   with machine-readable diagnostics on stderr and JSON results written to the
#   explicit output path
# - Required diagnostics: python/torch/cuda provenance, cache roots, patching
#   mode, and clear fatal errors when CUDA/tooling/model bootstrap assumptions fail

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl git build-essential ffmpeg unzip \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python --version \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

# Pin the CUDA torch stack first and do not let later installs replace it.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128

RUN python -m pip install --no-cache-dir --no-deps \
    "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"
RUN python -m pip install --no-cache-dir \
    neuralset neuraltrain==0.0.2 exca submitit mne scipy yt-dlp nilearn \
    x-transformers==1.27.20 einops timm transformers accelerate \
    pydantic omegaconf hydra-core julius soundfile "moviepy>=2.2.1" \
    hf_transfer spacy Levenshtein gtts langdetect wandb
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download en_core_web_lg

RUN python -m pip install --no-cache-dir \
    pyannote-audio==4.0.4 faster-whisper==1.2.1 ctranslate2==4.7.1 \
    nltk==3.9.4
RUN python -m pip install --no-cache-dir Cython packaging \
    && python -m pip install --no-cache-dir "nemo_toolkit[asr]"

RUN yt-dlp --version > /dev/null 2>&1 || { echo "FATAL: yt-dlp CLI not found"; exit 1; }

ENV HF_HOME=/models/hf \
    TRIBE_CACHE=/models/tribe \
    MNE_DATA=/models/mne_data \
    RARAMURI_SEED_CACHE_ROOT=/opt/raramuri-seed \
    RARAMURI_IMAGE_CONTRACT=docker/Dockerfile \
    RARAMURI_IMAGE_TARGETS=dgx-spark-gb10,runpod \
    RARAMURI_IMAGE_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
    RARAMURI_IMAGE_PYTHON=3.11 \
    RARAMURI_IMAGE_TORCH_CHANNEL=cu128 \
    RARAMURI_TRANSCRIPT_BACKEND=parakeet \
    RARAMURI_BUILD_AUTHORITY=local-canonical-dockerfile \
    RARAMURI_RUNPOD_BUILD_POLICY=bazel-derived-x64

RUN mkdir -p /models/tribe /models/hf /models/mne_data /opt/raramuri-seed /app

COPY docker/preload_assets.py /tmp/preload_assets.py
RUN export MNE_DATA=/opt/raramuri-seed/mne_data \
    && python /tmp/preload_assets.py \
    && rm /tmp/preload_assets.py

RUN python - <<'PY'
import os
import platform
import shutil
import torch

print('RARAMURI_IMAGE_READY')
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"hf_home={os.environ['HF_HOME']}")
print(f"tribe_cache={os.environ['TRIBE_CACHE']}")
print(f"mne_data={os.environ['MNE_DATA']}")
print(f"seed_cache_root={os.environ['RARAMURI_SEED_CACHE_ROOT']}")
print(f"transcript_backend={os.environ['RARAMURI_TRANSCRIPT_BACKEND']}")
print(f"ytdlp_path={shutil.which('yt-dlp')}")
print(f"build_authority={os.environ['RARAMURI_BUILD_AUTHORITY']}")
print(f"runpod_build_policy={os.environ['RARAMURI_RUNPOD_BUILD_POLICY']}")
PY

FROM base AS runtime

COPY docker/infer.py /app/infer.py
COPY docker/infer_server.py /app/infer_server.py

WORKDIR /app
ENTRYPOINT ["python", "/app/infer.py"]
