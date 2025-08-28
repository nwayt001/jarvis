#!/bin/bash

# JARVIS Launcher Script
# Supports both local and remote LLM/TTS configurations

# Default values
LLM_MODE="local"
TTS_MODE="local"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm-mode)
            LLM_MODE="$2"
            shift 2
            ;;
        --tts-mode)
            TTS_MODE="$2"
            shift 2
            ;;
        --remote)
            # Convenience option to run everything remotely
            LLM_MODE="remote"
            TTS_MODE="remote"
            shift
            ;;
        --local)
            # Convenience option to run everything locally
            LLM_MODE="local"
            TTS_MODE="local"
            shift
            ;;
        --help)
            echo "JARVIS Launcher Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --llm-mode <local|remote>  Set LLM server mode (default: local)"
            echo "  --tts-mode <local|remote>  Set TTS server mode (default: local)"
            echo "  --remote                    Run both LLM and TTS remotely"
            echo "  --local                     Run both LLM and TTS locally"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run everything locally"
            echo "  $0 --remote                 # Run everything on remote server"
            echo "  $0 --llm-mode remote        # Run LLM remotely, TTS locally"
            echo "  $0 --tts-mode remote        # Run TTS remotely, LLM locally"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate modes
if [[ "$LLM_MODE" != "local" && "$LLM_MODE" != "remote" ]]; then
    echo "Error: LLM_MODE must be 'local' or 'remote'"
    exit 1
fi

if [[ "$TTS_MODE" != "local" && "$TTS_MODE" != "remote" ]]; then
    echo "Error: TTS_MODE must be 'local' or 'remote'"
    exit 1
fi

# Export environment variables for jarvis.py
export JARVIS_LLM_MODE="$LLM_MODE"
export JARVIS_TTS_MODE="$TTS_MODE"

# Display configuration
echo "> Starting JARVIS with configuration:"
echo "   LLM Mode: $LLM_MODE"
echo "   TTS Mode: $TTS_MODE"
echo ""

# Check SSH connectivity if using remote mode
if [[ "$LLM_MODE" == "remote" || "$TTS_MODE" == "remote" ]]; then
    echo "= Checking SSH connectivity to 10.0.0.108..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes nicholas@10.0.0.108 "echo 'SSH connection successful'" > /dev/null 2>&1; then
        echo " SSH connection established"
    else
        echo "L Cannot connect to remote server (10.0.0.108)"
        echo "Please ensure:"
        echo "  1. The remote server is accessible"
        echo "  2. SSH key authentication is configured"
        echo "  3. You can connect with: ssh nicholas@10.0.0.108"
        exit 1
    fi
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "= Activating Python virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "= Activating Python virtual environment..."
    source ../venv/bin/activate
fi

# Run JARVIS
echo "=€ Launching JARVIS..."
echo "=" * 60
python jarvis.py

# Cleanup on exit
echo ""
echo "=K JARVIS shutdown complete"