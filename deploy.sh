#!/bin/bash
set -e

# IRST Library Production Deployment Script
# Usage: ./deploy.sh [environment] [version]

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "🚀 Deploying IRST Library to $ENVIRONMENT (version: $VERSION)"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed." >&2; exit 1; }

# Set environment variables
export IRST_ENV=$ENVIRONMENT
export IRST_VERSION=$VERSION
export COMPOSE_PROJECT_NAME=irst-library-$ENVIRONMENT

# Create necessary directories
mkdir -p logs data models checkpoints

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker-compose pull

# Start services
echo "🔧 Starting services..."
docker-compose up -d --remove-orphans

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Deployment successful!"
    echo "🌐 API available at: http://localhost:8000"
    echo "📊 Monitoring dashboard: http://localhost:3000"
    echo "📝 Logs: docker-compose logs -f"
else
    echo "❌ Health check failed!"
    echo "📝 Check logs: docker-compose logs"
    exit 1
fi

# Show running services
echo "🔍 Running services:"
docker-compose ps

echo "🎉 Deployment complete!"
