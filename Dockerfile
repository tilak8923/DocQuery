FROM python:3.12-slim

# Tesseract for OCR of scanned PDFs and images.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Docker hosts usually have the memory for the most accurate reader.
ENV DLQA_READER_MODEL=deepset/deberta-v3-large-squad2
ENV DLQA_TOP_K=5

# Pre-download the models so the container starts fast and can run offline.
RUN python -c "from dlqa.pipeline import DocumentQA; DocumentQA()"
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
