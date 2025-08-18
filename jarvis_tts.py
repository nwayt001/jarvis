"""
JARVIS TTS module with true streaming support using sounddevice
Real-time audio streaming for minimal latency
"""
import requests
import threading
import logging
import struct
import time
import sounddevice as sd

logger = logging.getLogger(__name__)

class JARVISStreamingTTS:
    """Handle text-to-speech for JARVIS with true streaming support"""
    
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
    
    def _parse_wav_header(self, header: bytes):
        """Parse WAV header to extract audio format info"""
        try:
            # Standard WAV header parsing
            # Offset 22: NumChannels (2 bytes)
            # Offset 24: SampleRate (4 bytes)  
            # Offset 34: BitsPerSample (2 bytes)
            channels = struct.unpack_from("<H", header, 22)[0]
            samplerate = struct.unpack_from("<I", header, 24)[0]
            bits_per_sample = struct.unpack_from("<H", header, 34)[0]
            
            # Map bits to numpy dtype for sounddevice
            dtype_map = {8: "int8", 16: "int16", 24: "int24", 32: "int32"}
            dtype = dtype_map.get(bits_per_sample, "int16")
            
            return channels, samplerate, dtype
        except Exception as e:
            logger.warning(f"Failed to parse WAV header: {e}, using defaults")
            return 1, 22050, "int16"  # Default mono 22kHz 16-bit
    
    def speak(self, text: str) -> None:
        """Stream TTS audio with minimal latency"""
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
                        "chunk_size": 25  # Small chunks for low latency
                    },
                    stream=True,
                    timeout=30
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
                
                # Split header and any leftover audio data
                leftover = header[44:]
                header = header[:44]
                
                # Parse WAV header
                channels, samplerate, dtype = self._parse_wav_header(header)
                logger.info(f"🔊 Streaming: {samplerate}Hz, {channels}ch, {dtype}")
                
                # Open sounddevice output stream
                with sd.RawOutputStream(
                    samplerate=samplerate,
                    channels=channels,
                    dtype=dtype,
                    blocksize=2048,
                    latency='low'
                ) as stream:
                    # Play leftover data from header
                    if leftover:
                        stream.write(leftover)
                    
                    # Stream remaining audio chunks in real-time
                    for chunk in iterator:
                        if self.stop_playback:
                            break
                        if chunk:
                            stream.write(chunk)
                
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


# Standalone streaming function for testing
def play_streaming_tts(tts_host: str, text: str):
    """Simple function to play streaming TTS"""
    response = requests.post(
        f"{tts_host}/tts/stream",
        json={
            "text": text,
            "exaggeration": 0.3,
            "cfg_weight": 0.5,
            "use_jarvis_voice": True,
            "stream": True,
            "chunk_size": 25
        },
        stream=True,
        timeout=30
    )
    
    response.raise_for_status()
    it = response.iter_content(chunk_size=16384)
    
    # Read WAV header (first 44 bytes)
    header = b""
    while len(header) < 44:
        header += next(it)
    
    header, leftover = header[:44], header[44:]
    
    # Parse header
    ch = struct.unpack_from("<H", header, 22)[0]
    sr = struct.unpack_from("<I", header, 24)[0]
    bps = struct.unpack_from("<H", header, 34)[0]
    dtype = {8: "int8", 16: "int16", 24: "int24", 32: "int32"}.get(bps, "int16")
    
    print(f"Audio: {ch}ch, {sr}Hz, {dtype}")
    
    # Open output stream and play
    with sd.RawOutputStream(samplerate=sr, channels=ch, dtype=dtype) as out:
        if leftover:
            out.write(leftover)
        for chunk in it:
            if chunk:
                out.write(chunk)


if __name__ == "__main__":
    # Test the streaming TTS
    tts_host = "http://10.0.0.108:8001"
    
    print("Testing JARVIS Streaming TTS...")
    print("-" * 40)
    
    tts = JARVISStreamingTTS(tts_host)
    
    test_text = "Good evening, Sir. This streaming synthesis provides minimal latency. You should hear my voice almost immediately."
    
    print(f"Speaking: {test_text}")
    tts.speak(test_text)
    tts.wait_for_speech()
    
    print("\nDone!")