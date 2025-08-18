"""
JARVIS TTS module with streaming support and click-free audio
Handles float32 WAV format and implements crossfading
"""
import requests
import threading
import logging
import struct
import time
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

class JARVISStreamingTTS:
    """Handle text-to-speech for JARVIS with click-free streaming"""
    
    def __init__(self, tts_host: str):
        self.tts_host = tts_host
        self.is_playing = False
        self.stop_playback = False
        
        # Check if streaming is available on server
        try:
            response = requests.get(f"{self.tts_host}/health", timeout=2)
            if response.status_code == 200:
                health = response.json()
                self.streaming_available = health.get('streaming_available', False)
                logger.info(f"✅ TTS Server connected - Streaming: {self.streaming_available}")
            else:
                self.streaming_available = False
        except:
            self.streaming_available = False
            logger.warning("⚠️ Could not connect to TTS server")
    
    def parse_wav_header(self, header: bytes):
        """Parse WAV header with proper format detection (PCM vs IEEE float)"""
        try:
            assert header[:4] == b'RIFF' and header[8:12] == b'WAVE', "Invalid WAV header"
            
            # Parse fmt chunk
            audio_format = struct.unpack_from('<H', header, 20)[0]   # 1=PCM, 3=IEEE float
            num_channels = struct.unpack_from('<H', header, 22)[0]
            sample_rate = struct.unpack_from('<I', header, 24)[0]
            bits_per_sample = struct.unpack_from('<H', header, 34)[0]
            
            # Determine dtype based on format
            if audio_format == 1:   # PCM
                dtype_map = {8: 'int8', 16: 'int16', 24: 'int24', 32: 'int32'}
                dtype = dtype_map.get(bits_per_sample, 'int16')
            elif audio_format == 3:  # IEEE float
                dtype = 'float32'
            else:
                logger.warning(f'Unsupported WAV format: {audio_format}, assuming float32')
                dtype = 'float32'
            
            logger.info(f"📊 WAV format: {audio_format} ({'float' if audio_format==3 else 'PCM'}), "
                       f"{num_channels}ch, {sample_rate}Hz, {bits_per_sample}bit → {dtype}")
            
            return audio_format, num_channels, sample_rate, bits_per_sample, dtype
        except Exception as e:
            logger.warning(f"Failed to parse WAV header: {e}, using defaults")
            return 3, 1, 22050, 32, 'float32'  # Default to float32 mono 22kHz
    
    def _bytes_to_frames(self, b: bytes, dtype: str, channels: int):
        """Convert byte stream to numpy array frames"""
        a = np.frombuffer(b, dtype=np.dtype(dtype).newbyteorder('<'))
        if channels > 1:
            a = a.reshape(-1, channels)
        else:
            a = a.reshape(-1, 1)
        return a
    
    def _frames_to_bytes(self, a: np.ndarray, dtype: str):
        """Convert numpy frames back to bytes"""
        return a.astype(np.dtype(dtype).newbyteorder('<'), copy=False).tobytes()
    
    def speak(self, text: str, overlap_ms: int = 6) -> None:
        """Stream TTS audio with crossfading to eliminate clicks"""
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
                        "chunk_size": 50  # Larger chunks for smoother crossfading
                    },
                    stream=True,
                    timeout=60
                )
                
                if response.status_code != 200:
                    logger.error(f"TTS request failed: {response.status_code}")
                    return
                
                response.raise_for_status()
                iterator = response.iter_content(chunk_size=16384)
                
                # Read WAV header (first 44 bytes)
                header = b""
                while len(header) < 44:
                    chunk = next(iterator)
                    if not chunk:
                        break
                    header += chunk
                
                if len(header) < 44:
                    logger.error("Incomplete WAV header")
                    return
                
                # Split header and leftover data
                leftover = header[44:]
                header = header[:44]
                
                # Parse WAV header
                fmt, channels, samplerate, bps, dtype = self.parse_wav_header(header)
                
                # Open sounddevice output stream
                logger.info(f"🔊 Starting playback: {samplerate}Hz, {channels}ch, {dtype}")
                
                with sd.RawOutputStream(
                    samplerate=samplerate,
                    channels=channels,
                    dtype=dtype,
                    blocksize=0,  # Let sounddevice choose
                    latency='low'
                ) as stream:
                    
                    # Crossfade setup
                    overlap = max(1, int(samplerate * overlap_ms / 1000.0))  # frames
                    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, None]
                    fade_out = 1.0 - fade_in
                    prev_tail = None
                    
                    def process_and_write(pcm_bytes):
                        """Process chunk with crossfading"""
                        nonlocal prev_tail
                        
                        if not pcm_bytes:
                            return
                        
                        frames = self._bytes_to_frames(pcm_bytes, dtype, channels)
                        
                        # Convert to float32 for processing
                        if dtype.startswith('int'):
                            # Scale integer PCM to [-1, 1]
                            scale = np.float32(1.0 / (2**(8*int(bps/8)-1)))
                            f = frames.astype(np.float32) * scale
                        else:
                            f = frames.astype(np.float32)
                        
                        # First chunk - no previous tail
                        if prev_tail is None:
                            if len(f) <= overlap:
                                # Very short first chunk - just save it
                                prev_tail = f.copy()
                                return
                            
                            # For first chunk, play everything INCLUDING overlap
                            # (no need to hold back since there's nothing to blend with yet)
                            # This fixes the pause after first chunk
                            
                            # Apply a short fade-in to avoid initial click
                            fade_in_len = min(overlap // 2, len(f))
                            if fade_in_len > 0:
                                fade = np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)[:, None]
                                f[:fade_in_len] *= fade
                            
                            # Save tail for next chunk
                            prev_tail = f[-overlap:].copy()
                            
                            # Play the entire first chunk
                            if dtype == 'float32':
                                write_data = np.clip(f, -1, 1)
                            else:
                                write_data = np.clip(f, -1, 1) * (2**(8*int(bps/8)-1)-1)
                            
                            stream.write(self._frames_to_bytes(write_data, dtype))
                            return
                        
                        # Subsequent chunks - crossfade with previous tail
                        if len(f) < overlap:
                            # Not enough to crossfade, accumulate
                            prev_tail = np.concatenate([prev_tail, f], axis=0)[-overlap:]
                            return
                        
                        # Crossfade the overlap region
                        head = f[:overlap]
                        blended = prev_tail[-overlap:] * fade_out + head * fade_in
                        
                        # Middle section (no crossfade needed)
                        if len(f) > 2 * overlap:
                            middle = f[overlap:-overlap]
                        else:
                            middle = np.empty((0, channels), np.float32)
                        
                        # Save new tail for next chunk
                        prev_tail = f[-overlap:].copy()
                        
                        # Combine blended and middle sections
                        out_frames = np.concatenate([blended, middle], axis=0) if len(middle) > 0 else blended
                        
                        # Convert back to original dtype
                        if dtype == 'float32':
                            write_frames = np.clip(out_frames, -1, 1)
                        else:
                            write_frames = np.clip(out_frames, -1, 1) * (2**(8*int(bps/8)-1)-1)
                        
                        stream.write(self._frames_to_bytes(write_frames, dtype))
                    
                    # Process leftover from header
                    if leftover:
                        process_and_write(leftover)
                    
                    # Process streaming chunks
                    for chunk in iterator:
                        if self.stop_playback:
                            break
                        if chunk:
                            process_and_write(chunk)
                    
                    # Flush remaining tail with fade-out to avoid final click
                    if prev_tail is not None and len(prev_tail) > 0:
                        n = len(prev_tail)
                        tail_fade = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
                        faded = prev_tail * tail_fade
                        
                        if dtype == 'float32':
                            write_frames = faded
                        else:
                            write_frames = np.clip(faded, -1, 1) * (2**(8*int(bps/8)-1)-1)
                        
                        stream.write(self._frames_to_bytes(write_frames, dtype))
                
                logger.info("✅ Playback completed")
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
            finally:
                self.is_playing = False
        
        # Start streaming in background thread
        audio_thread = threading.Thread(target=_stream_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def stop(self):
        """Stop current playback"""
        self.stop_playback = True
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            time.sleep(0.1)


# Standalone function with crossfading
def play_stream_with_crossfade(url, payload, overlap_ms=6):
    """Standalone streaming function with crossfading"""
    tts = JARVISStreamingTTS(url.rsplit('/', 1)[0])  # Extract base URL
    
    # Use the class method which handles everything
    tts.speak(payload.get('text', ''), overlap_ms)
    tts.wait_for_speech()


if __name__ == "__main__":
    # Test the click-free streaming
    tts_host = "http://10.0.0.108:8001"
    
    print("Testing JARVIS Streaming TTS (click-free)...")
    print("-" * 40)
    
    tts = JARVISStreamingTTS(tts_host)
    
    test_text = "Good evening, Sir. This enhanced streaming synthesis eliminates audio clicks through intelligent crossfading. The audio quality should be pristine."
    
    print(f"Speaking: {test_text}")
    tts.speak(test_text, overlap_ms=6)  # 6ms crossfade
    tts.wait_for_speech()
    
    print("\nDone!")