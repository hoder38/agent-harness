# §10 cross-encoder reranker (bge-reranker-v2-m3), CUDA torch (cu128: RTX 50xx / Blackwell needs CUDA ≥ 12.8).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Taipei HF_HOME=/hf
WORKDIR /srv
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
RUN useradd -u 1000 -m rerank && mkdir -p /hf && chown rerank:rerank /hf
USER rerank

EXPOSE 8300
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8300"]
