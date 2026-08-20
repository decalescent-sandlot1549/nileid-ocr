# ─────────────────────────────────────────────────────────────────────
# NileID — CPU-only image serving the HTTP API on port 7860.
#
# Model weights are NOT baked into the image: they are fetched at build
# time from the GitHub release. Pass --build-arg SKIP_MODELS=1 to build
# without them and mount /app/models at run time instead.
#
#   docker build -t nileid .
#   docker run --rm -p 7860:7860 nileid
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NILEID_MODEL_DIR=/app/models

# OpenCV and EasyOCR need these shared libraries at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (CPU wheels) in its own layer: large and rarely changes ──
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# ── Remaining dependencies ───────────────────────────────────────────
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ── Application ──────────────────────────────────────────────────────
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip install --no-deps -e .

# ── Model weights ────────────────────────────────────────────────────
ARG SKIP_MODELS=0
RUN if [ "$SKIP_MODELS" = "0" ]; then \
        python scripts/download_models.py --dest /app/models ; \
    else \
        mkdir -p /app/models ; \
    fi

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# EasyOCR caches its language models under the home directory.
ENV EASYOCR_MODULE_PATH=/home/appuser/.EasyOCR

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "nileid.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
