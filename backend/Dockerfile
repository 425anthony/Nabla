FROM python:3.11-slim

WORKDIR /app

# ca-certificates so the MNIST download over HTTPS validates at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app ./app

# MNIST cache lives at app/.mnist_cache (see app/data.py). Create it so it can
# be mounted as a volume to persist downloads; if empty it re-downloads cleanly
# on the first /train call.
RUN mkdir -p /app/app/.mnist_cache

EXPOSE 8000

# Honor $PORT when provided (e.g. Render); default to 8000 for local/compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
