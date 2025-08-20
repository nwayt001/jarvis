#!/bin/bash

# JARVIS Mac TTS Server Startup Script

echo "🎙️ Starting JARVIS Mac TTS Server..."
echo "🍎 Optimized for Apple Silicon with Metal Performance Shaders"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import chatterbox, sounddevice, torch, torchaudio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install chatterbox-tts sounddevice torch torchaudio fastapi uvicorn
fi

# Check if JARVIS voice sample exists
if [ ! -f "audio_library/jarvis_sample.wav" ] && [ ! -f "jarvis_sample.wav" ]; then
    echo "⚠️ Warning: JARVIS voice sample not found"
    echo "   The TTS will use the default voice"
    echo "   To use JARVIS voice, place a sample at: audio_library/jarvis_sample.wav"
fi

echo ""
echo "📡 Server will be available at: http://127.0.0.1:8001"
echo "   Health check: http://127.0.0.1:8001/health"
echo "   API docs: http://127.0.0.1:8001/docs"
echo ""

# Start the TTS server
python3 serve_tts_mac.py