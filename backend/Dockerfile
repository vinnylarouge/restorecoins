# HuggingFace Space (Docker SDK) — see backend/README.md for Space config.
# Built for ZeroGPU runtime; falls back to CPU gracefully if no accelerator.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    RESTORECOINS_MODE=real

WORKDIR /app

# System packages (libGL is required by opencv-python-headless on some bases).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-real.txt /app/
RUN pip install -r /app/requirements-real.txt

COPY . /app/

# HF Spaces inject the commit SHA as $SPACE_COMMIT; provenance.py picks it up.
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
