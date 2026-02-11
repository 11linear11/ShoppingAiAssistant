# ============================================================
# MCP Servers Dockerfile
# Shared image for MCP servers (interpret, search, embedding)
# The specific server is selected via CMD in docker-compose
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy MCP server code
COPY src/ ./src/

# Copy data files needed by servers
COPY BrandScore.json CategoryW.json full_category_embeddings.json ./

# Create logs directory
RUN mkdir -p /app/logs

# ---- Environment Variables ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Debug / Production mode
    DEBUG_MODE=false

# Default: no specific server (must override in docker-compose)
# Example overrides in docker-compose:
#   interpret:  CMD ["python", "-m", "src.mcp_servers.interpret_server"]
#   search:     CMD ["python", "-m", "src.mcp_servers.search_server"]
#   embedding:  CMD ["python", "-m", "src.mcp_servers.embedding_server"]
CMD ["python", "-m", "src.mcp_servers.run_servers"]
