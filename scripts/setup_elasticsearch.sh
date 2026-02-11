#!/bin/bash
# Setup Elasticsearch
# For Shopping AI Assistant

set -e

echo "🔧 Setting up Elasticsearch..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Configuration
ES_VERSION="8.12.0"
ES_PASSWORD="${ELASTICSEARCH_PASSWORD:-changeme}"
ES_CONTAINER_NAME="elasticsearch-shopping"

# Stop existing container if running
if docker ps -q -f name=$ES_CONTAINER_NAME &> /dev/null; then
    echo "⏹️  Stopping existing Elasticsearch container..."
    docker stop $ES_CONTAINER_NAME
fi

# Remove existing container if exists
if docker ps -aq -f name=$ES_CONTAINER_NAME &> /dev/null; then
    echo "🗑️  Removing existing Elasticsearch container..."
    docker rm $ES_CONTAINER_NAME
fi

# Pull image
echo "📥 Pulling Elasticsearch $ES_VERSION image..."
docker pull docker.elastic.co/elasticsearch/elasticsearch:$ES_VERSION

# Run Elasticsearch
echo "🚀 Starting Elasticsearch..."
docker run -d \
    --name $ES_CONTAINER_NAME \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    -v elasticsearch-data:/usr/share/elasticsearch/data \
    docker.elastic.co/elasticsearch/elasticsearch:$ES_VERSION

# Wait for Elasticsearch to be ready
echo "⏳ Waiting for Elasticsearch to be ready (this may take a minute)..."
for i in {1..60}; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# Test connection
if curl -s http://localhost:9200 | grep -q "cluster_name"; then
    echo "✅ Elasticsearch is running and ready!"
    echo ""
    echo "Connection details:"
    echo "  URL: http://localhost:9200"
    echo "  User: elastic (if security enabled)"
    echo ""

    # Create the index
    echo "📦 Creating shopping_products index..."
    curl -X PUT "http://localhost:9200/shopping_products" \
        -H "Content-Type: application/json" \
        -d '{
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "product_name": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "brand_name": {
                        "type": "text"
                    },
                    "category_name": {
                        "type": "keyword"
                    },
                    "price": {
                        "type": "long"
                    },
                    "discount_price": {
                        "type": "long"
                    },
                    "has_discount": {
                        "type": "boolean"
                    },
                    "discount_percentage": {
                        "type": "float"
                    },
                    "image_url": {
                        "type": "keyword"
                    },
                    "product_embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": true,
                        "similarity": "cosine"
                    }
                }
            }
        }' 2>/dev/null

    echo ""
    echo "✅ Index created successfully!"
    echo ""
    echo "To check Elasticsearch status:"
    echo "  curl http://localhost:9200/_cluster/health"
    echo ""
    echo "To stop Elasticsearch:"
    echo "  docker stop $ES_CONTAINER_NAME"
else
    echo "❌ Elasticsearch is not responding. Check the container logs:"
    echo "  docker logs $ES_CONTAINER_NAME"
    exit 1
fi
