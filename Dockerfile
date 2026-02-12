# Base Image: Lightweight Python
FROM python:3.10-slim as builder

# Set env vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps (gcc often needed for some python packages like numpy)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
# We remove 'torch' and 'transformers' if they are in requirements.txt but not needed for inference
# For this specific deployment, we are using the Baseline model (scikit-learn only).
# However, our requirements.txt might still have them. Ideally we'd have a requirements-prod.txt.
# We will use the existing requirements.txt for now but ideally prune it.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Application Code
COPY app ./app
COPY models ./models
# We don't need scripts or data in the production image

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
