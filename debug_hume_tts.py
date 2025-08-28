#!/usr/bin/env python3
"""
Debug script to identify why Hume TTS audio isn't playing
"""
import os
import asyncio
import base64
import pyaudio
import wave
import tempfile
from hume import AsyncHumeClient
from hume.tts import (
    FormatPcm,
    PostedUtterance,
    PostedUtteranceVoiceWithName,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def debug_hume_tts():
    """Debug Hume TTS with detailed diagnostics"""
    
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        print("ERROR: HUME_API_KEY not set")
        return
    
    print("=" * 60)
    print("Hume TTS Debug Diagnostic")
    print("=" * 60)
    
    # Initialize PyAudio
    print("\n1. Testing PyAudio...")
    audio = pyaudio.PyAudio()
    
    # List all audio devices
    print("\nAvailable audio devices:")
    default_output = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"   Device {i}: {info['name']} (Output channels: {info['maxOutputChannels']})")
            if info.get('index') == audio.get_default_output_device_info()['index']:
                default_output = info
                print(f"   ^^^ DEFAULT OUTPUT DEVICE ^^^")
    
    if not default_output:
        print("ERROR: No default output device found!")
        return
    
    print(f"\nUsing output device: {default_output['name']}")
    
    # Test with a simple tone first
    print("\n2. Testing PyAudio with a simple tone (you should hear a beep)...")
    import numpy as np
    
    duration = 1.0
    frequency = 440.0
    sample_rate = 48000
    
    samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)).astype(np.float32)
    samples = (samples * 0.3 * 32767).astype(np.int16)  # Convert to 16-bit PCM
    
    stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=sample_rate,
        output=True
    )
    
    print("   Playing test tone...")
    stream.write(samples.tobytes())
    stream.stop_stream()
    stream.close()
    print("   Did you hear the beep? (1 second, 440Hz)")
    
    # Now test Hume
    print("\n3. Testing Hume TTS...")
    hume = AsyncHumeClient(api_key=api_key)
    
    # Test text
    test_text = "Testing Hume audio playback. Can you hear me now?"
    print(f"   Text: '{test_text}'")
    
    # Prepare utterance with Jarvis voice
    voice = PostedUtteranceVoiceWithName(name="Jarvis")
    utterances = [PostedUtterance(text=test_text, voice=voice)]
    
    # Collect all audio chunks first
    print("\n4. Streaming from Hume...")
    all_audio_bytes = bytearray()
    chunk_count = 0
    
    stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=48000,
        output=True,
        frames_per_buffer=2048
    )
    
    try:
        async for snippet in hume.tts.synthesize_json_streaming(
            utterances=utterances,
            format=FormatPcm(type="pcm"),
            instant_mode=True
        ):
            audio_bytes = base64.b64decode(snippet.audio)
            chunk_count += 1
            
            print(f"   Chunk {chunk_count}: {len(audio_bytes)} bytes", end="")
            
            # Check if audio data looks valid
            if len(audio_bytes) > 0:
                # Check first few bytes
                first_bytes = audio_bytes[:10]
                print(f" - First bytes: {[hex(b) for b in first_bytes[:5]]}", end="")
                
                # Play immediately
                stream.write(audio_bytes)
                all_audio_bytes.extend(audio_bytes)
                print(" - PLAYED")
            else:
                print(" - EMPTY!")
    
    finally:
        stream.stop_stream()
        stream.close()
    
    print(f"\n   Total chunks: {chunk_count}")
    print(f"   Total bytes: {len(all_audio_bytes)}")
    
    # Save to file for inspection
    print("\n5. Saving audio to file for inspection...")
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    with wave.open(temp_filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(48000)
        wav_file.writeframes(bytes(all_audio_bytes))
    
    print(f"   Audio saved to: {temp_filename}")
    print("   You can play this file with any audio player to verify the content")
    
    # Try playing the saved file
    print("\n6. Playing back the saved file...")
    with wave.open(temp_filename, 'rb') as wav_file:
        data = wav_file.readframes(wav_file.getnframes())
        
        stream = audio.open(
            format=audio.get_format_from_width(wav_file.getsampwidth()),
            channels=wav_file.getnchannels(),
            rate=wav_file.getframerate(),
            output=True
        )
        
        stream.write(data)
        stream.stop_stream()
        stream.close()
    
    print("   Playback complete")
    
    # Test non-streaming for comparison
    print("\n7. Testing non-streaming synthesis...")
    result = await hume.tts.synthesize_json(
        utterances=utterances,
        format=FormatPcm(type="pcm")
    )
    
    if result and result.generations:
        audio_data = base64.b64decode(result.generations[0].audio)
        print(f"   Generated {len(audio_data)} bytes")
        
        # Play non-streaming audio
        stream = audio.open(
            format=audio.get_format_from_width(2),
            channels=1,
            rate=48000,
            output=True
        )
        
        print("   Playing non-streaming audio...")
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
        print("   Done")
    
    audio.terminate()
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("\nWhat you should have heard:")
    print("1. A test beep (440Hz tone)")
    print("2. Streaming TTS: 'Testing Hume audio playback. Can you hear me now?'")
    print("3. The same audio played from saved file")
    print("4. Non-streaming TTS of the same text")
    print(f"\nCheck the saved file: {temp_filename}")

if __name__ == "__main__":
    asyncio.run(debug_hume_tts())