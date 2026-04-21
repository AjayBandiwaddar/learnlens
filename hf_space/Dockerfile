# NumberSort OpenEnv environment for LearnLens demo
# HF Space: https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort

# python:3.11-slim -- DO NOT use bookworm digest variants (build failures on HF)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY app.py .
COPY models.py .
COPY number_sort_environment.py .

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# workers=1 mandatory -- WebSocket sessions hold environment state
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]