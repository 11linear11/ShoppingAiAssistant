# ============================================================
# Frontend Dockerfile (React + Vite → Nginx)
# Multi-stage: build static assets, serve with nginx
# ============================================================

# --- Stage 1: Build ---
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first (better caching)
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --legacy-peer-deps 2>/dev/null || npm install --legacy-peer-deps

# Copy source code
COPY frontend/ .

# Set API URL for production build
ARG VITE_API_URL=http://localhost:8080
ENV VITE_API_URL=${VITE_API_URL}

# Build the app
RUN npm run build


# --- Stage 2: Serve with Nginx ---
FROM nginx:alpine AS runtime

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx template (rendered at container startup via envsubst)
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:3000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
