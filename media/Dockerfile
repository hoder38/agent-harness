# §5 media-worker: ffmpeg, faster-whisper (CUDA fp16 / CPU int8), OpenCC, RapidOCR (PP-OCRv6), Docling (CPU torch).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Taipei \
    HF_HOME=/models/hf OMP_NUM_THREADS=8
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tzdata libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# PP-OCRv6 det/rec (one model: Simplified, Traditional, Japanese, English) baked in: no download at run time
RUN python -c "\
from rapidocr import RapidOCR, OCRVersion, ModelType as M; v = OCRVersion.PPOCRV6; \
RapidOCR(params={'Det.ocr_version': v, 'Rec.ocr_version': v, 'Det.model_type': M.SMALL, 'Rec.model_type': M.SMALL})"
# ctranslate2 loads cuBLAS/cuDNN from the pip wheels
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
RUN useradd -u 1000 -m media && mkdir -p /models && chown media:media /models

FROM base AS test
RUN pip install pytest pytest-asyncio fakeredis
USER media
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

FROM base AS runtime
COPY --chown=media:media media ./media
USER media
CMD ["arq", "media.worker.Settings"]
