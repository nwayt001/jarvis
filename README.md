# JARVIS - Just A Rather Very Intelligent System

An AI assistant with Tony Stark's JARVIS personality, powered by Ollama and LangGraph, with chatterbox text-to-speech.

## Features

- **JARVIS Personality**: British butler-like AI assistant with dry wit and professional demeanor
- **Multiple Tools**: Web search, weather forecasts, file operations, bash commands, and system monitoring
- **Text-to-Speech**: Optional Chatterbox TTS server for voice output
- **Remote LLM**: Runs on a separate Linux machine with GPU acceleration
- **LangGraph Agent**: Sophisticated tool-calling and decision-making capabilities

## Architecture

```
Mac (Frontend)          Linux Box (10.0.0.108)
├── Jarvis.py     →     ├── Ollama Server (port 11434)
└── venv/               ├── TTS Server (port 8001)
                        └── Web Scraper Monitor
```

## Installation

### Prerequisites
- Python 3.12+
- SSH access to Linux box (for scraper monitoring)
- Ollama installed on Linux machine

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd jarvis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# or manually:
pip install langchain-ollama langgraph pygame requests googlesearch-python beautifulsoup4
```

4. Configure Ollama on Linux box:
```bash
# On Linux machine (10.0.0.108)
ollama pull gpt-oss:20b  # or your preferred model
ollama serve
```

5. (Optional) Start TTS server on Linux box:
```bash
python serve_tts.py
```

## Configuration

Edit the configuration section in `Jarvis.py`:

```python
OLLAMA_HOST = "http://10.0.0.108:11434"  # Your Linux machine IP
TTS_HOST = "http://10.0.0.108:8001"      # TTS server
MODEL_NAME = "gpt-oss:20b"               # Your Ollama model
ENABLE_TTS = True                         # Toggle voice output
```

## Usage

Start JARVIS:
```bash
python Jarvis.py
```

Example commands:
- "What's the weather in Baltimore tomorrow evening?"
- "Search for news about ChatGPT-5"
- "Check if the web scraper is running"
- "List files in this directory"
- "Create a Python script that calculates fibonacci numbers"
- "What's the system load on the Linux box?"

## Available Tools

- **web_search**: Google/DuckDuckGo search with fallbacks
- **get_weather**: Current conditions and 3-day forecast
- **execute_bash**: Run shell commands (with safety checks)
- **read_file**: Read file contents
- **write_file**: Create/modify files (auto-backup)
- **list_directory**: Browse directories
- **check_web_scraper_status**: Monitor Red-Dot-Scraper process

## TTS Server

The optional TTS server (`serve_tts.py`) provides GLaDOS-style voice synthesis:
- Runs on GPU-enabled Linux machine
- Chatterbox TTS model
- Automatic text chunking for long responses
- REST API on port 8001

## Security Notes

- File operations exclude sensitive paths (.ssh, .env, etc.)
- Dangerous commands are blocked (rm -rf /, mkfs, etc.)
- System directories are protected from writes
- Automatic backups before file modifications

## Troubleshooting

### Connection Issues
- Verify Ollama is running: `curl http://10.0.0.108:11434/api/tags`

### TTS Issues
- Reduce MAX_TEXT_LENGTH in serve_tts.py if crashes occur
- Check GPU memory on Linux box
- Disable TTS: Set `ENABLE_TTS = False`

### Search Issues
- Google search rate limits: Add delays or use SerpAPI
- Set SERPAPI_KEY environment variable for better results

## License

MIT

## Author

Nicholas - AI Agent Development