#!/bin/bash

# Test script for the API

set -e

API_URL="${API_URL:-http://localhost:8000}"

echo "🧪 Testing LLaMA LoRA Inference API at $API_URL"
echo ""

# Test 1: Root endpoint
echo "Test 1: Root endpoint"
curl -s "$API_URL/" | python3 -m json.tool
echo ""

# Test 2: Health check
echo "Test 2: Health check"
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# Test 3: Model info
echo "Test 3: Model information"
curl -s "$API_URL/model-info" | python3 -m json.tool
echo ""

# Test 4: Text generation
echo "Test 4: Text generation"
curl -s -X POST "$API_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is HeapTrace Technology?",
    "max_new_tokens": 100,
    "temperature": 1
  }' | python3 -m json.tool
echo ""

echo "✅ All tests completed!"
