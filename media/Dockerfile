# §5 media-worker: ffmpeg, faster-whisper (CUDA fp16 / CPU int8), OpenCC, RapidOCR (PP-OCRv6, onnxruntime-gpu),
# Docling (CUDA torch). cu128: the RTX 50xx (Blackwell, sm_120) needs CUDA >= 12.8. GPU first, CPU fallback (2026-11-10).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Taipei \
    HF_HOME=/models/hf OMP_NUM_THREADS=8
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tzdata libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip uninstall -y onnxruntime && pip install onnxruntime-gpu==1.26.0
# onnxruntime-gpu (same module name as onnxruntime: only one may be installed). 1.26.0 is the last PyPI build for
# CUDA 12; 1.27+ need CUDA 13 + a cu13 cuDNN that clashes with torch cu128's libcudnn.so.9. It has no sm_120 kernels:
# the first CUDA run JIT-compiles PTX (~30 s), cached in CUDA_CACHE_PATH on the models volume.
# PP-OCRv6 det/rec (one model: Simplified, Traditional, Japanese, English) baked in: no download at run time
RUN python -c "\
from rapidocr import RapidOCR, OCRVersion, ModelType as M; v = OCRVersion.PPOCRV6; \
RapidOCR(params={'Det.ocr_version': v, 'Rec.ocr_version': v, 'Det.model_type': M.SMALL, 'Rec.model_type': M.SMALL})"
# ctranslate2 and onnxruntime-gpu load the CUDA 12 libraries from torch's nvidia pip wheels
ARG NV=/usr/local/lib/python3.12/site-packages/nvidia
ENV LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cudnn/lib:$NV/curand/lib:$NV/cufft/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib \
    CUDA_CACHE_PATH=/models/cuda-cache CUDA_CACHE_MAXSIZE=1073741824
RUN useradd -u 1000 -m media && mkdir -p /models && chown media:media /models

FROM base AS test
RUN pip install pytest pytest-asyncio fakeredis
USER media
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

FROM base AS runtime
COPY --chown=media:media media ./media
USER media
CMD ["arq", "media.worker.Settings"]
