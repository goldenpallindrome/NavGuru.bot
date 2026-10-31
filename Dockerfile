FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libgomp1 build-essential && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_lg
COPY . .
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/data/pdfs && chown -R rag:rag /app
USER rag
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Optional offline evaluation image; keeps RAGAS dependencies out of the live API image.
FROM app AS evaluation
USER root
COPY requirements-eval.txt .
RUN pip install --no-cache-dir -r requirements-eval.txt
USER rag
