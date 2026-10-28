# §12b read-only page renderer. Chromium + deps come with the pinned Playwright image (keep versions in sync).
FROM mcr.microsoft.com/playwright/python:v1.64.1-noble

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=/tmp TZ=Asia/Taipei
WORKDIR /srv
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt
COPY app.py .
USER pwuser

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
