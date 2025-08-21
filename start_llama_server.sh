#!/bin/bash

# JARVIS Local LLM Server Startup Script
# Edit the MODEL_PATH to point to your GGUF model file

MODEL_PATH="${1:-/path/to/gpt-oss-20b-mxfp4.gguf}"
PORT="${2:-8080}"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "Usage: $0 <path-to-gguf-model> [port]"
    echo "Example: $0 ~/models/gpt-oss-20b-mxfp4.gguf 8080"
    exit 1
fi

echo "🚀 Starting llama-server with model: $MODEL_PATH"
echo "📡 Server will be available at: http://localhost:$PORT"
echo ""
echo "Configuration:"
echo "  - Flash Attention: Enabled"
echo "  - GPU Layers: 99"
echo "  - Context: Unlimited"
echo "  - Jinja Templates: Enabled"
echo ""

# Start llama-server with the specified configuration
llama-server \
    --model "$MODEL_PATH" \
    -c 0 \
    -fa \
    --jinja \
    --reasoning-format deepseek \
    -ngl 99 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-disable