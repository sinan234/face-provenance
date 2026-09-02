# Face detection + web provenance + blockchain verification — CLI image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application source and assets.
COPY src ./src
COPY contracts ./contracts
COPY scripts ./scripts
COPY data ./data
COPY .env.example ./

# Demo/test images live on a mounted volume; keep the default demo fixture.
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["process", "/app/data/input.jpg", "--mode", "demo", "--chain", "memory"]
