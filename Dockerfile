# Build using Python 3.11 to support advanced scikit-learn serialization layers
FROM python:3.11-slim

# Enforce hardware compute thread restrictions to prevent CPU usage spikes
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install compilation prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build isolated storage directory matrices
RUN mkdir -p /app/storage/benchmarks /app/storage/uploads /app/storage/training_outputs

# Install library layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bind core asset paths into image workspace
COPY v1_L/ /app/v1_L/
COPY v1_NB/ /app/v1_NB/
COPY v1_T/ /app/v1_T/
COPY data.csv /app/data.csv
COPY production_deployment/ /app/production_deployment/

EXPOSE 8000
WORKDIR /app/production_deployment

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]