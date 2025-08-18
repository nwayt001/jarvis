#!/bin/bash
# Install script for streaming TTS support

echo "🚀 Installing chatterbox-streaming for lower latency TTS..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install the streaming package
pip install chatterbox-streaming

# Check if installation was successful
python -c "from chatterbox_streaming.tts import ChatterboxTTS; print('✅ chatterbox-streaming installed successfully')" 2>/dev/null || {
    echo "⚠️  chatterbox-streaming installation failed or not compatible"
    echo "The server will fall back to standard chatterbox"
}

echo ""
echo "📝 To use the new streaming TTS server:"
echo "1. On your Linux box, run: python serve_tts_streaming.py"
echo "2. The server will auto-detect streaming capability"
echo "3. JARVIS will automatically use streaming if available"
echo ""
echo "🎯 Benefits of streaming:"
echo "- Lower latency to first audio"
echo "- Audio starts playing before generation completes"
echo "- Better perceived responsiveness"