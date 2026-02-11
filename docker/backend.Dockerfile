# ============================================================
# Backend Dockerfile (FastAPI + Agent)
# Multi-stage build for smaller production image
# ============================================================

# --- Stage 1: Builder (install dependencies) ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/

# Copy data files needed at runtime
COPY BrandScore.json CategoryW.json full_category_embeddings.json ./

# Create logs directory
RUN mkdir -p /app/logs

# ---- Environment Variables ----
# These can be overridden by docker-compose
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # App
    HOST=0.0.0.0 \
    PORT=8080 \
    # Debug / Production mode
    DEBUG_MODE=false \
    DEBUG=false \
    # Timeouts
    AGENT_TIMEOUT=120

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health', timeout=5)" || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
