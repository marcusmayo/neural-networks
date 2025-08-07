FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
RUN mkdir -p mlruns models
COPY mlruns/ ./mlruns/

EXPOSE 1234
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:1234/health || exit 1
CMD ["mlflow", "models", "serve", "-m", "models:/BikeRentalModel/latest", "--host", "0.0.0.0", "-p", "1234", "--no-conda"]
