# §10 cross-encoder reranker (bge-reranker-v2-m3), CPU-only torch.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Taipei HF_HOME=/hf
WORKDIR /srv
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
RUN useradd -u 1000 -m rerank && mkdir -p /hf && chown rerank:rerank /hf
USER rerank

EXPOSE 8300
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8300"]
