#!/usr/bin/env python3
"""
Chatterbox TTS Server for GLaDOS
Run with: python serve_chatterbox.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import io
from pydantic import BaseModel
import logging
from typing import Optional, List
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GLaDOS TTS Server")

# Global model variable
model = None
GLADOS_VOICE_PATH = "jarvis_sample.wav"  # Update this path

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.0  # GLaDOS dramatic style
    cfg_weight: float = 0.6    # Deliberate pacing
    use_glados_voice: bool = True

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        logger.info("Loading Chatterbox model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        logger.info(f"Model loaded successfully on: {device}")
        
        # Check if GLaDOS voice sample exists
        if os.path.exists(GLADOS_VOICE_PATH):
            logger.info(f"GLaDOS voice sample found at: {GLADOS_VOICE_PATH}")
        else:
            logger.warning(f"GLaDOS voice sample not found at: {GLADOS_VOICE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def split_text_into_chunks(text: str, max_length: int = 200) -> List[str]:
    """Split text into chunks at sentence boundaries"""
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

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text and return audio stream"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        text_length = len(request.text)
        logger.info(f"Generating speech for text of length {text_length}: {request.text[:50]}...")
        
        # Check if text needs to be chunked
        MAX_TEXT_LENGTH = 200  # Adjust based on what works best
        
        if text_length > MAX_TEXT_LENGTH:
            logger.info(f"Text too long ({text_length} chars), splitting into chunks...")
            chunks = split_text_into_chunks(request.text, MAX_TEXT_LENGTH)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Generate audio for each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                
                if request.use_glados_voice and os.path.exists(GLADOS_VOICE_PATH):
                    wav = model.generate(
                        chunk, 
                        audio_prompt_path=GLADOS_VOICE_PATH,
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
            
            # Concatenate all audio chunks with small pauses
            silence_duration = int(model.sr * 0.3)  # 0.3 second pause between chunks
            silence = torch.zeros(1, silence_duration)
            
            combined_wav = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                combined_wav = torch.cat([combined_wav, silence, chunk], dim=1)
            
            wav = combined_wav
        else:
            # Process normally for short text
            if request.use_glados_voice and os.path.exists(GLADOS_VOICE_PATH):
                wav = model.generate(
                    request.text, 
                    audio_prompt_path=GLADOS_VOICE_PATH,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            else:
                logger.warning("Using default voice (GLaDOS voice not found or disabled)")
                wav = model.generate(
                    request.text,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
        
        # Convert to audio stream in memory
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        logger.info("Audio generation complete")
        
        # Return as streaming response
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=glados_speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is running and model is loaded"""
    return {
        "status": "operational" if model is not None else "loading",
        "device": str(model.device) if model else "not loaded",
        "glados_voice_available": os.path.exists(GLADOS_VOICE_PATH)
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "GLaDOS TTS Server",
        "endpoints": {
            "/tts": "POST - Generate speech from text",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server
    logger.info("Starting GLaDOS TTS Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8001,
        log_level="info"
    )