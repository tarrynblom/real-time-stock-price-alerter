#!/bin/bash

# Simple deployment script for Stock Price Alerter

set -e

echo "🚀 Deploying Stock Price Alerter..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.template .env
    echo "⚠️  Please update .env with your API keys before running!"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Build and start the application
echo "🏗️  Building and starting application..."
docker-compose up -d --build

# Wait for health check
echo "⏳ Waiting for application to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health &>/dev/null; then
        echo "✅ Application is ready!"
        break
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "🌐 Stock Alerter is running:"
echo "  📈 API: http://localhost:8000"
echo "  📚 Docs: http://localhost:8000/docs"
echo "  🔍 Health: http://localhost:8000/health"
echo ""
echo "💡 Commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop: docker-compose down" 