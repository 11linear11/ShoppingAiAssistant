# ============================================================
# Frontend Development Dockerfile (Vite dev server)
# Used only with docker-compose.dev.yml
# ============================================================

FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --legacy-peer-deps 2>/dev/null || npm install --legacy-peer-deps

# Copy source (will be overridden by volume mount in dev)
COPY frontend/ .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
