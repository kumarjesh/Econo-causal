# ===================================================================
# EconoCausal: Multi-Stage Enterprise Production Dockerfile
# Serves FastAPI backend (Port 8000) and Streamlit dashboard (Port 8501)
# ===================================================================

# Stage 1: Build & Dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python scientific packages & dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime shared libraries (OpenMP for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project application codebase and data artifacts
COPY . /app/

# Environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT_API=8000 \
    PORT_DASHBOARD=8501

# Expose microservice and dashboard ports
EXPOSE 8000 8501

# Healthcheck probe verifying API readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default startup command (runs FastAPI microservice)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

