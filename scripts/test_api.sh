#!/bin/bash

echo "🧪 Testing Stock Alerter API..."

# Wait for server to start
sleep 5

# Test endpoints
echo "Testing health endpoint..."
curl -X GET "http://localhost:8000/health" | jq

echo "Testing training endpoint..."
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL"}' | jq

# Wait for training
sleep 30

echo "Testing prediction endpoint..."
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL"}' | jq

echo "Testing alert endpoint..."
curl -X POST "http://localhost:8000/alert" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL"}' | jq 