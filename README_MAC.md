# JARVIS Mac Local Edition

This version of JARVIS runs entirely on your Mac using a local LLM via llama.cpp.

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

## Running JARVIS

### Option 1: Manual Setup (Recommended for testing)

1. Start the llama-server in one terminal:
```bash
./start_llama_server.sh /path/to/your/model.gguf
```

2. In another terminal, run JARVIS:
```bash
python Jarvis_mac.py
```

### Option 2: Automatic Setup

Just run JARVIS and let it start the server for you:
```bash
python Jarvis_mac.py
```
When prompted, choose 'y' to start the server and provide the path to your GGUF model.

## Configuration

Edit `Jarvis_mac.py` to modify:
- `MODEL_PATH`: Default path to your GGUF model
- `LLAMA_CPP_HOST`: LLM server endpoint (default: http://localhost:8080)
- `ENABLE_TTS`: Set to True if you have local TTS configured

## Key Differences from Linux Version

1. **Local LLM**: Uses llama.cpp with flash attention instead of remote Ollama
2. **OpenAI-compatible API**: Uses langchain-openai to connect to llama-server's OpenAI-compatible endpoint
3. **No TTS by default**: TTS is disabled since it was running on the Linux box
4. **Performance optimizations**: Flash attention enabled for better performance

## Troubleshooting

If the LLM server doesn't start:
1. Make sure llama.cpp is installed: `brew install llama.cpp`
2. Check that your GGUF model file path is correct
3. Ensure port 8080 is not in use
4. Try running the server manually with: `./start_llama_server.sh /path/to/model.gguf`

## Notes

- The server uses flash attention (`-fa`) for improved performance
- Context is unlimited (`-c 0`) to handle long conversations
- All 99 GPU layers are used (`-ngl 99`) for maximum acceleration
- Jinja templating is enabled for proper prompt formatting