FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only PyTorch is much smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

# Download the embedding model at build time so startup is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

# The API key is NOT stored in the image; pass it with `docker run -e`.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
