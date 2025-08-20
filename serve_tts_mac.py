#!/usr/bin/env python3
"""
Mac-optimized Chatterbox TTS Server for JARVIS
Uses Metal Performance Shaders (MPS) for Apple Silicon acceleration
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torchaudio as ta
import io
from pydantic import BaseModel
import logging
from typing import Optional, AsyncGenerator
import asyncio
import numpy as np
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Mac TTS Server")

# Global model variable
model = None
JARVIS_VOICE_PATH = "audio_library/jarvis_sample.wav"

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.3  # JARVIS measured style
    cfg_weight: float = 0.5    # Deliberate pacing
    use_jarvis_voice: bool = True
    stream: bool = True        # Enable streaming by default

@app.on_event("startup")
async def load_model():
    """Load the model on startup with Mac optimizations"""
    global model,  JARVIS_VOICE_PATH
    try:
        logger.info("Loading Chatterbox model for Mac...")
        
        # Follow the EXACT pattern from the Mac example
        # Detect device (Mac with M1/M2/M3/M4)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        map_location = torch.device(device)
        
        if device == "mps":
            logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
        else:
            logger.info("💻 Using CPU (MPS not available)")
        
        # Patch torch.load EXACTLY as shown in the example
        torch_load_original = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch_load_original(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Now import and load model AFTER patching
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        
        logger.info(f"✅ Model loaded successfully on: {device}")
        
        # Check if JARVIS voice sample exists
        import os
        print(f"Checking for JARVIS voice sample at: {JARVIS_VOICE_PATH}")
        if os.path.exists(JARVIS_VOICE_PATH):
            logger.info(f"✅ JARVIS voice sample found at: {JARVIS_VOICE_PATH}")
        else:
            # Try alternate paths
            alt_paths = [
                "audio_library/jarvis_sample.aif",
                "jarvis_sample.wav",
                "jarvis_sample.aif"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    JARVIS_VOICE_PATH = alt_path
                    logger.info(f"✅ JARVIS voice found at: {JARVIS_VOICE_PATH}")
                    break
            else:
                logger.warning(f"⚠️ JARVIS voice sample not found, will use default voice")
                
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

async def generate_audio_stream(text: str, request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """Generate audio with streaming support"""
    try:
        import os
        
        # Check if streaming is available
        if hasattr(model, 'generate_stream'):
            logger.info(f"Starting streaming generation for: {text[:50]}...")
            
            generator_params = {
                "text": text,
                "exaggeration": request.exaggeration,
                "cfg_weight": request.cfg_weight,
            }
            
            # Add voice if available
            if request.use_jarvis_voice and os.path.exists(JARVIS_VOICE_PATH):
                generator_params["audio_prompt_path"] = JARVIS_VOICE_PATH
            
            # Generate with streaming
            wav_header_sent = False
            
            for audio_chunk, metrics in model.generate_stream(**generator_params):
                # Apply fade-in to first chunk
                if not wav_header_sent:
                    # Apply smooth fade-in to prevent clicks
                    fadein_samples = int(model.sr * 0.010)  # 10ms
                    if audio_chunk.shape[1] > fadein_samples:
                        t = torch.linspace(0, 1, fadein_samples, device=audio_chunk.device)
                        ramp = (3 * t**2 - 2 * t**3).unsqueeze(0)
                        audio_chunk[:, :fadein_samples] *= ramp
                    
                    # Add small silence pad for startup
                    silence = torch.zeros(1, int(model.sr * 0.005), dtype=audio_chunk.dtype, device=audio_chunk.device)
                    audio_chunk = torch.cat([silence, audio_chunk], dim=1)
                
                # Convert to float32 PCM
                pcm = audio_chunk.squeeze(0).detach().cpu().to(torch.float32).clamp(-1, 1)
                raw_bytes = pcm.numpy().astype('<f4').tobytes()
                
                # Send WAV header with first chunk
                if not wav_header_sent:
                    wav_header_sent = True
                    
                    # Create float32 WAV header
                    sample_rate = model.sr
                    channels = 1
                    bits_per_sample = 32
                    byte_rate = sample_rate * channels * 4
                    block_align = channels * 4
                    
                    header = b'RIFF'
                    header += b'\xff\xff\xff\xff'  # File size (streaming)
                    header += b'WAVE'
                    header += b'fmt '
                    header += struct.pack('<I', 16)  # fmt chunk size
                    header += struct.pack('<H', 3)   # Format: IEEE float
                    header += struct.pack('<H', channels)
                    header += struct.pack('<I', sample_rate)
                    header += struct.pack('<I', byte_rate)
                    header += struct.pack('<H', block_align)
                    header += struct.pack('<H', bits_per_sample)
                    header += b'data'
                    header += b'\xff\xff\xff\xff'  # Data size (streaming)
                    
                    yield header + raw_bytes
                else:
                    yield raw_bytes
                
                # Allow other async operations
                await asyncio.sleep(0)
                
        else:
            # Fallback to non-streaming generation
            logger.info("Using non-streaming generation")
            
            if request.use_jarvis_voice and os.path.exists(JARVIS_VOICE_PATH):
                wav = model.generate(
                    text,
                    audio_prompt_path=JARVIS_VOICE_PATH,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            else:
                wav = model.generate(
                    text,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            ta.save(buffer, wav, model.sr, format="wav")
            buffer.seek(0)
            yield buffer.read()
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"TTS request - Text length: {len(request.text)}, Stream: {request.stream}")
        
        if request.stream:
            # Stream audio as it's generated
            return StreamingResponse(
                generate_audio_stream(request.text, request),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=jarvis_speech.wav",
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # Non-streaming generation
            import os
            
            if request.use_jarvis_voice and os.path.exists(JARVIS_VOICE_PATH):
                wav = model.generate(
                    request.text,
                    audio_prompt_path=JARVIS_VOICE_PATH,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            else:
                wav = model.generate(
                    request.text,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            
            # Convert to audio stream
            buffer = io.BytesIO()
            ta.save(buffer, wav, model.sr, format="wav")
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=jarvis_speech.wav"}
            )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """Dedicated streaming endpoint"""
    request.stream = True
    return await text_to_speech(request)

@app.get("/health")
async def health_check():
    """Check if the service is running and model is loaded"""
    import os
    has_streaming = hasattr(model, 'generate_stream') if model else False
    
    # Get device info safely
    device_info = "not loaded"
    if model and hasattr(model, 'device'):
        device_str = str(model.device)
        if 'mps' in device_str.lower():
            device_info = "Apple Metal (MPS) 🍎"
        elif 'cuda' in device_str.lower():
            device_info = "NVIDIA CUDA 🎮"
        elif 'cpu' in device_str.lower():
            device_info = "CPU 💻"
        else:
            device_info = device_str
    
    return {
        "status": "operational" if model is not None else "loading",
        "device": device_info,
        "jarvis_voice_available": os.path.exists(JARVIS_VOICE_PATH) if JARVIS_VOICE_PATH else False,
        "streaming_available": has_streaming,
        "platform": "macOS"
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "JARVIS Mac TTS Server",
        "endpoints": {
            "/tts": "POST - Generate speech",
            "/tts/stream": "POST - Stream speech",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "optimizations": {
            "device": "Apple Metal (MPS) when available",
            "streaming": "Low-latency audio streaming",
            "voice_cloning": "JARVIS voice support"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting JARVIS Mac TTS Server...")
    logger.info("🍎 Optimized for Apple Silicon")
    
    # Run on localhost for better performance
    uvicorn.run(
        app,
        host="127.0.0.1",  # Local only for security
        port=8001,
        log_level="info"
    )