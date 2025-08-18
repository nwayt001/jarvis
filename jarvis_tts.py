"""
Enhanced JARVIS TTS module with streaming support
Drop-in replacement for the JARVISTTS class with streaming capabilities
"""
import pygame
import requests
import io
import threading
import logging
import queue
import time

logger = logging.getLogger(__name__)
ENABLE_TTS = True  # Set to False to disable TTS

# Streaming TTS class for JARVIS
class JARVISStreamingTTS:
    """Handle text-to-speech for JARVIS with streaming support"""
    
    def __init__(self, tts_host: str, enable_streaming: bool = True):
        self.tts_host = tts_host
        self.is_playing = False
        self.enable_streaming = enable_streaming
        self.audio_queue = queue.Queue()
        self.stop_playback = False
        
        # Check if streaming is available
        try:
            response = requests.get(f"{self.tts_host}/health", timeout=2)
            if response.status_code == 200:
                health = response.json()
                self.streaming_available = health.get('streaming_available', False)
                if self.streaming_available:
                    logger.info("✅ Streaming TTS available")
                else:
                    logger.info("⚠️ Streaming not available, using standard TTS")
            else:
                self.streaming_available = False
        except:
            self.streaming_available = False
            
    def speak_streaming(self, text: str) -> None:
        """Generate and play JARVIS speech with streaming"""
        if not text:
            return
            
        def _stream_audio():
            try:
                self.is_playing = True
                self.stop_playback = False
                
                # Request streaming TTS
                response = requests.post(
                    f"{self.tts_host}/tts/stream",
                    json={
                        "text": text,
                        "exaggeration": 0.3,
                        "cfg_weight": 0.5,
                        "use_jarvis_voice": True,
                        "stream": True,
                        "chunk_size": 25  # Smaller chunks for lower latency
                    },
                    stream=True,
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Accumulate streaming chunks
                    audio_buffer = io.BytesIO()
                    first_chunk = True
                    chunk_count = 0
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if self.stop_playback:
                            break
                            
                        if chunk:
                            chunk_count += 1
                            
                            # For first chunk, include WAV header
                            if first_chunk:
                                audio_buffer.write(chunk)
                                first_chunk = False
                                
                                # Start playing after getting first chunk
                                if chunk_count == 1 and len(chunk) > 44:  # WAV header is 44 bytes
                                    audio_buffer.seek(0)
                                    try:
                                        sound = pygame.mixer.Sound(audio_buffer)
                                        sound.play()
                                        logger.info(f"🔊 Started playback (streaming)")
                                    except:
                                        pass  # Continue accumulating
                            else:
                                # Append subsequent chunks (without WAV headers)
                                audio_buffer.write(chunk)
                    
                    # If we couldn't start streaming playback, play the complete audio
                    if chunk_count > 0 and not pygame.mixer.get_busy():
                        audio_buffer.seek(0)
                        sound = pygame.mixer.Sound(audio_buffer)
                        sound.play()
                        logger.info(f"🔊 Playing complete audio ({chunk_count} chunks)")
                    
                    # Wait for playback to finish
                    while pygame.mixer.get_busy() and not self.stop_playback:
                        pygame.time.wait(100)
                else:
                    logger.error(f"Streaming TTS failed: {response.status_code}")
                    # Fallback to non-streaming
                    self._play_audio_standard(text)
                    
            except requests.exceptions.ChunkedEncodingError:
                # This can happen with streaming, try to play what we have
                if audio_buffer.tell() > 44:
                    audio_buffer.seek(0)
                    sound = pygame.mixer.Sound(audio_buffer)
                    sound.play()
                    while pygame.mixer.get_busy() and not self.stop_playback:
                        pygame.time.wait(100)
            except Exception as e:
                logger.error(f"Streaming playback error: {e}")
                # Fallback to non-streaming
                self._play_audio_standard(text)
            finally:
                self.is_playing = False
        
        # Play audio in background thread
        audio_thread = threading.Thread(target=_stream_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def _play_audio_standard(self, text: str) -> None:
        """Fallback to standard non-streaming TTS"""
        try:
            response = requests.post(
                f"{self.tts_host}/tts",
                json={
                    "text": text,
                    "exaggeration": 0.3,
                    "cfg_weight": 0.5,
                    "use_jarvis_voice": True,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                audio_data = io.BytesIO(response.content)
                sound = pygame.mixer.Sound(audio_data)
                sound.play()
                
                while pygame.mixer.get_busy() and not self.stop_playback:
                    pygame.time.wait(100)
            else:
                logger.error(f"Standard TTS failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Standard TTS error: {e}")
    
    def speak(self, text: str) -> None:
        """Generate and play JARVIS speech (auto-selects streaming if available)"""
        if not text:
            return
            
        if self.enable_streaming and self.streaming_available:
            self.speak_streaming(text)
        else:
            # Use standard TTS
            def _play():
                self.is_playing = True
                self._play_audio_standard(text)
                self.is_playing = False
            
            audio_thread = threading.Thread(target=_play)
            audio_thread.daemon = True
            audio_thread.start()
    
    def stop(self):
        """Stop current playback"""
        self.stop_playback = True
        pygame.mixer.stop()
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            pygame.time.wait(100)

# standard TTS class from chatterbox
class JARVISTTS:
    """Handle text-to-speech for JARVIS"""
    
    def __init__(self, tts_host: str):
        self.tts_host = tts_host
        self.is_playing = False
        
    def speak(self, text: str) -> None:
        """Generate and play JARVIS speech"""
        if not ENABLE_TTS:
            return
            
        def _play_audio():
            try:
                self.is_playing = True
                
                # Request TTS
                response = requests.post(
                    f"{self.tts_host}/tts",
                    json={
                        "text": text,
                        "exaggeration": 0.3,  # More measured for JARVIS
                        "cfg_weight": 0.5,    # Balanced pacing
                        "use_glados_voice": False  # Use default or JARVIS voice if available
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Load audio from response
                    audio_data = io.BytesIO(response.content)
                    sound = pygame.mixer.Sound(audio_data)
                    
                    # Play and wait for completion
                    sound.play()
                    while pygame.mixer.get_busy():
                        pygame.time.wait(100)
                else:
                    logger.error(f"TTS request failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                self.is_playing = False
        
        # Play audio in background thread to not block
        audio_thread = threading.Thread(target=_play_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            pygame.time.wait(100)

# Example usage and testing
if __name__ == "__main__":
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Test with your TTS server
    tts = JARVISStreamingTTS("http://10.0.0.108:8001")
    
    # Test streaming
    print("Testing streaming TTS...")
    tts.speak("Good evening, Sir. All systems are now online. The streaming synthesis should provide lower latency.")
    tts.wait_for_speech()
    
    print("Done!")