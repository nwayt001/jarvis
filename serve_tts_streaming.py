#!/usr/bin/env python3
"""
Streaming Chatterbox TTS Server for JARVIS
Reduces latency by streaming audio chunks as they're generated
Run with: python serve_tts_streaming.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torchaudio as ta
import io
from pydantic import BaseModel
import logging
from typing import Optional, List, AsyncGenerator
import asyncio
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Streaming TTS Server")

# Global model variable
model = None
JARVIS_VOICE_PATH = "jarvis_sample.wav"  # Update this path

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.0  # JARVIS measured style
    cfg_weight: float = 0.6    # Deliberate pacing
    use_jarvis_voice: bool = True
    chunk_size: int = 25       # Smaller chunks for lower latency
    stream: bool = True        # Enable streaming by default

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        logger.info("Loading Chatterbox Streaming model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to import streaming version first
        try:
            from chatterbox_streaming.tts import ChatterboxTTS
            logger.info("Using chatterbox-streaming package")
        except ImportError:
            logger.warning("chatterbox-streaming not found, falling back to regular chatterbox")
            from chatterbox.tts import ChatterboxTTS
            
        model = ChatterboxTTS.from_pretrained(device=device)
        logger.info(f"Model loaded successfully on: {device}")
        
        # Check if JARVIS voice sample exists
        if os.path.exists(JARVIS_VOICE_PATH):
            logger.info(f"JARVIS voice sample found at: {JARVIS_VOICE_PATH}")
        else:
            logger.warning(f"JARVIS voice sample not found at: {JARVIS_VOICE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def split_text_into_chunks(text: str, max_length: int = 150) -> List[str]:
    """Split text into chunks at sentence boundaries for better streaming"""
    # First, try to split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is too long, split it by commas or semicolons
        if len(sentence) > max_length:
            sub_parts = re.split(r'[,;]\s*', sentence)
            for part in sub_parts:
                if len(current_chunk) + len(part) + 1 <= max_length:
                    current_chunk = current_chunk + " " + part if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
        elif len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def generate_audio_stream(text: str, request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """Generate audio in streaming chunks"""
    try:
        # Check if we have the streaming method
        if hasattr(model, 'generate_stream'):
            logger.info(f"Starting streaming generation for: {text[:50]}...")
            
            # Use streaming generation
            first_chunk = True
            audio_chunks = []
            
            generator_params = {
                "text": text,
                "exaggeration": request.exaggeration,
                "cfg_weight": request.cfg_weight,
                "chunk_size": request.chunk_size
            }
            
            if request.use_jarvis_voice and os.path.exists(JARVIS_VOICE_PATH):
                generator_params["audio_prompt_path"] = JARVIS_VOICE_PATH
            
            for audio_chunk, metrics in model.generate_stream(**generator_params):
                # Log first chunk latency
                if first_chunk and metrics.latency_to_first_chunk:
                    logger.info(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
                    first_chunk = False
                
                # Convert chunk to WAV format and yield
                chunk_buffer = io.BytesIO()
                ta.save(chunk_buffer, audio_chunk, model.sr, format="wav")
                chunk_buffer.seek(0)
                
                # Yield the audio data (skip WAV header for subsequent chunks)
                if len(audio_chunks) == 0:
                    yield chunk_buffer.read()  # First chunk includes header
                else:
                    chunk_buffer.seek(44)  # Skip WAV header for subsequent chunks
                    yield chunk_buffer.read()
                
                audio_chunks.append(audio_chunk)
                
                # Allow other async operations
                await asyncio.sleep(0)
                
        else:
            # Fallback to non-streaming generation
            logger.info("Streaming not available, using standard generation")
            
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
            
            # Convert to WAV and yield as single chunk
            buffer = io.BytesIO()
            ta.save(buffer, wav, model.sr, format="wav")
            buffer.seek(0)
            yield buffer.read()
            
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        raise

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text with optional streaming"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        text_length = len(request.text)
        logger.info(f"TTS request - Length: {text_length}, Streaming: {request.stream}")
        
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
            # Non-streaming fallback (for compatibility)
            MAX_TEXT_LENGTH = 200
            
            if text_length > MAX_TEXT_LENGTH:
                logger.info(f"Text too long ({text_length} chars), splitting into chunks...")
                chunks = split_text_into_chunks(request.text, MAX_TEXT_LENGTH)
                logger.info(f"Split into {len(chunks)} chunks")
                
                # Generate audio for each chunk
                audio_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    if request.use_jarvis_voice and os.path.exists(JARVIS_VOICE_PATH):
                        wav = model.generate(
                            chunk,
                            audio_prompt_path=JARVIS_VOICE_PATH,
                            exaggeration=request.exaggeration,
                            cfg_weight=request.cfg_weight
                        )
                    else:
                        wav = model.generate(
                            chunk,
                            exaggeration=request.exaggeration,
                            cfg_weight=request.cfg_weight
                        )
                    audio_chunks.append(wav)
                
                # Concatenate with pauses
                silence_duration = int(model.sr * 0.3)
                silence = torch.zeros(1, silence_duration)
                
                combined_wav = audio_chunks[0]
                for chunk in audio_chunks[1:]:
                    combined_wav = torch.cat([combined_wav, silence, chunk], dim=1)
                
                wav = combined_wav
            else:
                # Single generation for short text
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
            
            logger.info("Audio generation complete")
            
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
    """Dedicated streaming endpoint for real-time TTS"""
    request.stream = True  # Force streaming
    return await text_to_speech(request)

@app.get("/health")
async def health_check():
    """Check if the service is running and model is loaded"""
    has_streaming = hasattr(model, 'generate_stream') if model else False
    return {
        "status": "operational" if model is not None else "loading",
        "device": str(model.device) if model else "not loaded",
        "jarvis_voice_available": os.path.exists(JARVIS_VOICE_PATH),
        "streaming_available": has_streaming
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "JARVIS Streaming TTS Server",
        "endpoints": {
            "/tts": "POST - Generate speech (streaming optional)",
            "/tts/stream": "POST - Force streaming generation",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "features": {
            "streaming": "Low-latency audio streaming",
            "chunking": "Automatic text chunking for long inputs",
            "voice_cloning": "Custom voice support"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting JARVIS Streaming TTS Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )