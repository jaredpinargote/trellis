# Base Image: Lightweight Python
FROM python:3.10-slim as builder

# Set env vars to suppress warnings and bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies (gcc for numpy/spacy if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies from production list
COPY requirements-prod.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-prod.txt

# Download Spacy model for Presidio (en_core_web_sm is ~12MB)
RUN python -m spacy download en_core_web_sm

# Final Stage: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder to keep image small
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Scale-down: We don't need gcc in runtime
# Copy Application Code
COPY app ./app
COPY models/baseline.joblib ./models/baseline.joblib

# Create non-root user for security
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run Application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
