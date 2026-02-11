#!/bin/bash
# Setup Redis with RediSearch module
# For Shopping AI Assistant

set -e

echo "🔧 Setting up Redis with RediSearch..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name=redis-stack &> /dev/null; then
    echo "⏹️  Stopping existing redis-stack container..."
    docker stop redis-stack
fi

# Remove existing container if exists
if docker ps -aq -f name=redis-stack &> /dev/null; then
    echo "🗑️  Removing existing redis-stack container..."
    docker rm redis-stack
fi

# Pull latest image
echo "📥 Pulling redis-stack-server image..."
docker pull redis/redis-stack-server:latest

# Run Redis Stack
echo "🚀 Starting Redis Stack..."
docker run -d \
    --name redis-stack \
    -p 6379:6379 \
    -v redis-stack-data:/data \
    redis/redis-stack-server:latest

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
sleep 5

# Test connection
if docker exec redis-stack redis-cli ping | grep -q PONG; then
    echo "✅ Redis is running and ready!"
    echo ""
    echo "Connection details:"
    echo "  Host: localhost"
    echo "  Port: 6379"
    echo ""
    echo "To check Redis status:"
    echo "  docker exec redis-stack redis-cli INFO"
    echo ""
    echo "To stop Redis:"
    echo "  docker stop redis-stack"
else
    echo "❌ Redis is not responding. Check the container logs:"
    echo "  docker logs redis-stack"
    exit 1
fi
