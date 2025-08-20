# JARVIS Mac Local Edition

This version of JARVIS runs entirely on your Mac using a local LLM via llama.cpp and local TTS with Chatterbox.

## Prerequisites

1. Install llama.cpp (if not already installed):
```bash
brew install llama.cpp
```

2. Install Python dependencies:
```bash
pip install -r requirements_mac.txt
```

3. Download your GGUF model file (e.g., gpt-oss-20b-mxfp4.gguf)

4. (Optional) Add a JARVIS voice sample for voice cloning:
   - Place a WAV file at: `audio_library/jarvis_sample.wav`
   - Should be a clear recording of the voice you want to clone

## Running JARVIS

### Full Setup (LLM + TTS + JARVIS)

You'll need three terminal windows:

1. **Terminal 1 - Start the LLM server:**
```bash
./start_llama_server.sh /path/to/your/model.gguf
```

2. **Terminal 2 - Start the TTS server:**
```bash
./start_tts_mac.sh
```

3. **Terminal 3 - Run JARVIS:**
```bash
python Jarvis_mac.py
```

### Quick Setup (Auto-start LLM)

1. Start the TTS server:
```bash
./start_tts_mac.sh
```

2. Run JARVIS (will prompt to auto-start LLM):
```bash
python Jarvis_mac.py
```
When prompted, choose 'y' to start the LLM server and provide the path to your GGUF model.

## Configuration

Edit `Jarvis_mac.py` to modify:
- `MODEL_PATH`: Default path to your GGUF model
- `LLAMA_CPP_HOST`: LLM server endpoint (default: http://localhost:8080)
- `ENABLE_TTS`: Set to True if you have local TTS configured

## Key Differences from Linux Version

1. **Local LLM**: Uses llama.cpp with flash attention instead of remote Ollama
2. **Local TTS**: Chatterbox TTS runs locally on Mac with Metal acceleration
3. **OpenAI-compatible API**: Uses langchain-openai to connect to llama-server's OpenAI-compatible endpoint
4. **Performance optimizations**: 
   - Flash attention enabled for LLM
   - Metal Performance Shaders (MPS) for TTS on Apple Silicon
   - Streaming audio with crossfading for smooth playback

## Troubleshooting

### LLM Server Issues:
1. Make sure llama.cpp is installed: `brew install llama.cpp`
2. Check that your GGUF model file path is correct
3. Ensure port 8080 is not in use
4. Try running the server manually with: `./start_llama_server.sh /path/to/model.gguf`

### TTS Server Issues:
1. Install dependencies: `pip install chatterbox-tts sounddevice torch torchaudio`
2. Check port 8001 is not in use: `lsof -i :8001`
3. Test the server health: `curl http://127.0.0.1:8001/health`
4. For Apple Silicon Macs, MPS should be automatically detected
5. If you see "Internal Server Error", restart the TTS server

## Notes

- The server uses flash attention (`-fa`) for improved performance
- Context is unlimited (`-c 0`) to handle long conversations
- All 99 GPU layers are used (`-ngl 99`) for maximum acceleration
- Jinja templating is enabled for proper prompt formatting