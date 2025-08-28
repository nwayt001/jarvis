"""
JARVIS Hume TTS module with streaming support
Uses Hume AI's text-to-speech API for high-quality voice synthesis
"""
import os
import asyncio
import base64
import threading
import logging
import time
from typing import Optional, Protocol
from contextlib import contextmanager
import pyaudio
from hume import AsyncHumeClient
from hume.tts import (
    FormatPcm,
    PostedContextWithGenerationId,
    PostedUtterance,
    PostedUtteranceVoiceWithName,
)

logger = logging.getLogger(__name__)

class JARVISHumeTTS:
    """Handle text-to-speech for JARVIS using Hume AI"""
    
    def __init__(self, api_key: Optional[str] = None, voice_name: Optional[str] = "Jarvis2"):
        """
        Initialize Hume TTS
        
        Args:
            api_key: Hume API key (can also be set via HUME_API_KEY env var)
            voice_name: Optional cloned voice name (requires Creator plan). 
                       If None, uses voice description instead.
        """
        self.api_key = api_key or os.getenv("HUME_API_KEY")
        if not self.api_key:
            raise EnvironmentError("HUME_API_KEY not found. Please set it in environment or pass it to constructor.")
        
        self.is_playing = False
        self.stop_playback = False
        self.hume_client = None
        self.voice_name = voice_name  # Use cloned voice if provided (requires Creator plan)
        self.voice_description = "A refined, sophisticated British AI assistant with a calm, professional demeanor, similar to JARVIS from Iron Man"
        self.generation_id = None
        self.audio = None
        self.stream = None
        self._event_loop = None
        self._loop_thread = None
        
        # Initialize PyAudio for playback
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("✅ PyAudio initialized for Hume TTS playback")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            raise
        
        # Initialize Hume client in a persistent event loop
        self._init_event_loop()
        self._init_hume_client()
    
    def _init_event_loop(self):
        """Initialize a persistent event loop in a background thread"""
        def run_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        # Give the loop time to start
        time.sleep(0.1)
    
    def _init_hume_client(self):
        """Initialize Hume client using existing 'Jarvis' voice"""
        try:
            # Run initialization in the persistent event loop
            future = asyncio.run_coroutine_threadsafe(
                self._init_hume_async(),
                self._event_loop
            )
            future.result(timeout=5)  # Wait for initialization
        except Exception as e:
            logger.error(f"Failed to initialize Hume: {e}")
            raise
    
    async def _init_hume_async(self):
        """Initialize Hume client async"""
        try:
            self.hume_client = AsyncHumeClient(api_key=self.api_key)
            
            if self.voice_name:
                logger.info(f"✅ Hume TTS initialized using cloned voice: {self.voice_name}")
            else:
                logger.info("✅ Hume TTS initialized using voice description (no cloned voice)")
                # Optionally create an initial voice to establish the generation_id
                try:
                    result = await self.hume_client.tts.synthesize_json(
                        utterances=[
                            PostedUtterance(
                                description=self.voice_description,
                                text="Systems initialized.",
                            )
                        ]
                    )
                    if result and result.generations:
                        self.generation_id = result.generations[0].generation_id
                        logger.info("Initial voice generation completed")
                except Exception as e:
                    logger.debug(f"Could not create initial voice: {e}")
                    # Not critical - we can still use description for each request
            
        except Exception as e:
            logger.error(f"Failed to initialize Hume client: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure Hume is initialized before use"""
        if not self.hume_client:
            raise RuntimeError("Hume TTS not properly initialized")
        if not self._event_loop or not self._event_loop.is_running():
            raise RuntimeError("Event loop not running")
    
    
    def speak(self, text: str, overlap_ms: int = 6) -> None:
        """Generate and play TTS audio using Hume AI (non-streaming)"""
        if not text:
            return
        
        # Ensure Hume is initialized
        self._ensure_initialized()
        
        # Wait for any previous playback to finish
        self.wait_for_speech()
        
        self.is_playing = True
        self.stop_playback = False
        
        # Start a thread to handle the async playback
        def run_tts():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the non-streaming TTS
                loop.run_until_complete(self._generate_and_play(text))
                
            except Exception as e:
                logger.error(f"Error during Hume TTS playback: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.is_playing = False
        
        # Start playback in background thread
        playback_thread = threading.Thread(target=run_tts, daemon=True)
        playback_thread.start()
        
        # Give it a moment to start
        time.sleep(0.2)
    
    async def _generate_and_play(self, text: str):
        """Generate audio using non-streaming API and play it"""
        stream = None
        try:
            # Create a fresh Hume client for this thread
            hume = AsyncHumeClient(api_key=self.api_key)
            
            # Prepare utterances
            if self.voice_name:
                voice = PostedUtteranceVoiceWithName(name=self.voice_name)
                utterances = [PostedUtterance(text=text, voice=voice)]
            else:
                utterances = [PostedUtterance(text=text, description=self.voice_description)]
            
            context = PostedContextWithGenerationId(generation_id=self.generation_id) if self.generation_id else None
            
            logger.info(f"Generating audio for: '{text[:50]}...'")
            
            # Use non-streaming synthesis
            result = await hume.tts.synthesize_json(
                context=context,
                utterances=utterances,
                format=FormatPcm(type="pcm")
            )
            
            if result and result.generations and not self.stop_playback:
                # Get the audio data
                audio_bytes = base64.b64decode(result.generations[0].audio)
                logger.info(f"Generated {len(audio_bytes)} bytes of audio")
                
                # Update generation_id for continuity
                if hasattr(result.generations[0], 'generation_id'):
                    self.generation_id = result.generations[0].generation_id
                
                # Open audio stream and play
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(2),
                    channels=1,
                    rate=48000,
                    output=True,
                    frames_per_buffer=4096,
                )
                
                # Play the audio
                stream.write(audio_bytes)
                logger.info("✅ Audio playback completed")
            else:
                logger.warning("No audio generated from Hume")
                    
        except Exception as e:
            logger.error(f"Hume TTS error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
    
    def stop(self):
        """Stop current playback"""
        self.stop_playback = True
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            time.sleep(0.1)
    
    def __del__(self):
        """Clean up resources on deletion"""
        # Stop the event loop
        #if self._event_loop and self._event_loop.is_running():
        #    self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        
        # Clean up PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass


# Alias for compatibility with existing code
#JARVISStreamingTTS = JARVISHumeTTS


if __name__ == "__main__":
    # Test the Hume streaming TTS
    import sys
    
    print("Testing JARVIS Hume TTS...")
    print("-" * 40)
    
    # Check for API key
    if not os.getenv("HUME_API_KEY"):
        print("ERROR: Please set HUME_API_KEY environment variable")
        sys.exit(1)
    
    # Check if user wants to use a cloned voice
    voice_name = None
    if len(sys.argv) > 1:
        voice_name = sys.argv[1]
        print(f"Using cloned voice: {voice_name}")
    else:
        print("Using voice description (no cloned voice)")
        print("Tip: Pass voice name as argument to use cloned voice: python jarvis_hume_tts.py Jarvis")
    
    try:
        tts = JARVISHumeTTS(voice_name=voice_name)
        
        # Give it a moment to initialize
        time.sleep(1)
        
        test_text = "Good evening, Sir. This is the Hume AI voice synthesis system using your cloned Jarvis voice."
        
        print(f"Speaking: {test_text}")
        tts.speak(test_text)
        tts.wait_for_speech()
        
        print("\nTesting multiple utterances...")
        
        texts = [
            "System diagnostics complete.",
            "All parameters are within normal ranges.",
            "Shall I proceed with the requested operation?"
        ]
        
        for text in texts:
            print(f"Speaking: {text}")
            tts.speak(text)
            tts.wait_for_speech()
            time.sleep(0.5)
        
        print("\nDone!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)