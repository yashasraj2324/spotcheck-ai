# API image only. The model checkpoint is mounted or copied in at deploy time, never baked into git.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[api]"

ENV SKIN_LESION_CHECKPOINT=/models/best.pth
EXPOSE 8000
CMD ["uvicorn", "skin_lesion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
