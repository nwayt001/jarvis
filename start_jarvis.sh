#!/bin/bash
# Start JARVIS with local TTS server on Mac

echo "🚀 Starting JARVIS System..."
echo "================================"

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Shutting down JARVIS systems..."
    
    # Kill TTS server if running
    if [ ! -z "$TTS_PID" ]; then
        echo "   Stopping TTS server (PID: $TTS_PID)..."
        kill $TTS_PID 2>/dev/null
    fi
    
    # Kill any remaining Python processes related to JARVIS
    pkill -f "serve_tts_mac.py" 2>/dev/null
    
    echo "✅ JARVIS shutdown complete"
    exit 0
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS"
    echo "   TTS performance may be reduced on other platforms"
fi

# Check for required Python packages
echo "🔍 Checking dependencies..."
python3 -c "import chatterbox" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Chatterbox not installed!"
    echo "   Please run: pip install chatterbox-tts"
    exit 1
fi

python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not installed!"
    echo "   Please run: pip install torch torchaudio"
    exit 1
fi

# Check if MPS is available (Apple Silicon)
echo "🍎 Checking for Apple Silicon acceleration..."
python3 -c "import torch; print('   ✅ Metal Performance Shaders available!' if torch.backends.mps.is_available() else '   ℹ️  Using CPU (MPS not available)')"

# Start TTS server in background
echo -e "\n📢 Starting TTS server..."
python3 serve_tts_mac.py > tts_server.log 2>&1 &
TTS_PID=$!

# Wait for TTS server to start
echo "   Waiting for TTS server to initialize..."
for i in {1..30}; do
    curl -s http://127.0.0.1:8001/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ TTS server ready!"
        break
    fi
    sleep 1
    
    # Check if process is still running
    if ! kill -0 $TTS_PID 2>/dev/null; then
        echo "   ❌ TTS server failed to start!"
        echo "   Check tts_server.log for details"
        exit 1
    fi
    
    if [ $i -eq 30 ]; then
        echo "   ⚠️  TTS server slow to start, proceeding anyway..."
    fi
done

# Display TTS server status
echo "   TTS Server PID: $TTS_PID"
curl -s http://127.0.0.1:8001/health | python3 -m json.tool 2>/dev/null || echo "   (Unable to get server status)"

# Start JARVIS
echo -e "\n🤖 Starting JARVIS..."
echo "================================"
echo ""

# Run JARVIS Mac version (this will block until user exits)
python3 Jarvis_mac.py

# Cleanup will be called automatically on exit