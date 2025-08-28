#!/usr/bin/env python3
"""
Debug script for Hume TTS audio playback
"""
import os
import sys
import time
import asyncio
import base64
import pyaudio
from hume import AsyncHumeClient
from hume.tts import (
    FormatPcm,
    PostedUtterance,
    PostedUtteranceVoiceWithName,
)
from dotenv import load_dotenv
# Load the environment variables from the .env file
load_dotenv()
async def test_hume_audio(use_cloned_voice=True):
    """Test Hume audio generation and playback"""
    
    # Check for API key
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        print("ERROR: Please set HUME_API_KEY environment variable")
        return False
    
    voice_name = "Jarvis" if use_cloned_voice else None
    
    print("1. Initializing Hume client...")
    hume = AsyncHumeClient(api_key=api_key)
    
    print("2. Testing PyAudio...")
    audio = pyaudio.PyAudio()
    
    # List audio devices
    print("\nAvailable audio devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"   Device {i}: {info['name']} (Output channels: {info['maxOutputChannels']})")
    
    # Get default output device
    default_output = audio.get_default_output_device_info()
    print(f"\nDefault output device: {default_output['name']}")
    
    print("\n3. Generating speech with Hume...")
    
    # First, let's try a simple non-streaming test
    print("   Testing non-streaming synthesis first...")
    try:
        if voice_name:
            print(f"   Using cloned voice: {voice_name}")
            utterances = [
                PostedUtterance(
                    text="Testing audio playback. Can you hear me?",
                    voice=PostedUtteranceVoiceWithName(name=voice_name)
                )
            ]
        else:
            print("   Using voice description (no cloned voice)")
            utterances = [
                PostedUtterance(
                    text="Testing audio playback. Can you hear me?",
                    description="A refined, sophisticated British AI assistant voice"
                )
            ]
        
        result = await hume.tts.synthesize_json(
            utterances=utterances,
            format=FormatPcm(type="pcm")
        )
        
        if result and result.generations:
            print(f"   ✅ Generated audio (length: {len(result.generations[0].audio)} chars)")
            
            # Decode and play
            audio_bytes = base64.b64decode(result.generations[0].audio)
            print(f"   Decoded audio bytes: {len(audio_bytes)} bytes")
            
            # Play with PyAudio
            print("   Playing audio...")
            stream = audio.open(
                format=audio.get_format_from_width(2),  # 16-bit
                channels=1,
                rate=48000,
                output=True
            )
            
            stream.write(audio_bytes)
            stream.stop_stream()
            stream.close()
            print("   ✅ Playback complete")
            
        else:
            print("   ❌ No audio generated")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Now test streaming
    print("\n4. Testing streaming synthesis...")
    try:
        stream = audio.open(
            format=audio.get_format_from_width(2),  # 16-bit
            channels=1,
            rate=48000,
            output=True
        )
        
        print("   Streaming audio...")
        chunk_count = 0
        total_bytes = 0
        
        if voice_name:
            stream_utterances = [
                PostedUtterance(
                    text="Good evening, Sir. This is a streaming test of the cloned voice.",
                    voice=PostedUtteranceVoiceWithName(name=voice_name)
                )
            ]
        else:
            stream_utterances = [
                PostedUtterance(
                    text="Good evening, Sir. This is a streaming test using voice description.",
                    description="A refined, sophisticated British AI assistant voice"
                )
            ]
        
        # Instant mode requires a voice, so disable it when using description
        instant_mode = voice_name is not None
        
        async for snippet in hume.tts.synthesize_json_streaming(
            utterances=stream_utterances,
            format=FormatPcm(type="pcm"),
            instant_mode=instant_mode
        ):
            audio_bytes = base64.b64decode(snippet.audio)
            chunk_count += 1
            total_bytes += len(audio_bytes)
            print(f"   Chunk {chunk_count}: {len(audio_bytes)} bytes", end="\r")
            stream.write(audio_bytes)
        
        print(f"\n   ✅ Streamed {chunk_count} chunks, {total_bytes} total bytes")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"   ❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    audio.terminate()
    return True

def main():
    print("=" * 60)
    print("Hume TTS Audio Debug Test")
    print("=" * 60)
    
    # Check if user wants to test with cloned voice
    use_cloned = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--cloned":
        use_cloned = True
        print("Testing with cloned voice (requires Creator plan)\n")
    else:
        print("Testing with voice description (works on all plans)")
        print("Use --cloned flag to test with cloned voice\n")
    use_cloned = True  # Force using cloned voice for testing purposes
    # Run the async test
    success = asyncio.run(test_hume_audio(use_cloned_voice=use_cloned))
    
    if success:
        print("\n✅ All audio tests completed successfully!")
        print("\nIf you didn't hear anything, check:")
        print("1. Your system volume")
        print("2. The correct output device is selected")
        print("3. PyAudio is using the right device")
    else:
        print("\n❌ Audio tests failed")

if __name__ == "__main__":
    main()