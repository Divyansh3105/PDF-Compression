FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY compress.py .
COPY templates templates
COPY assets assets
EXPOSE 8000
# One process: job state is in memory. Threads handle concurrency.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:8000", "compress:app"]
