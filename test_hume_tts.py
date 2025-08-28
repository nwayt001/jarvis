#!/usr/bin/env python3
"""
Quick test script for Hume TTS integration
"""
import os
import sys
import time

# Check for API key
if not os.getenv("HUME_API_KEY"):
    print("ERROR: Please set HUME_API_KEY environment variable")
    print("Example: export HUME_API_KEY='your-api-key-here'")
    sys.exit(1)

from jarvis_hume_tts import JARVISHumeTTS

def main():
    print("=" * 60)
    print("Testing Hume TTS Integration for JARVIS")
    print("Using existing 'Jarvis' voice clone")
    print("=" * 60)
    
    try:
        # Initialize Hume TTS
        print("\n1. Initializing Hume TTS with 'Jarvis' voice...")
        tts = JARVISHumeTTS()
        
        # Wait for initialization
        print("   Waiting for initialization...")
        time.sleep(1)
        
        # Test basic speech
        print("\n2. Testing basic speech...")
        test_text = "Good evening, Sir. This is a test of the Hume text-to-speech integration."
        print(f"   Speaking: '{test_text}'")
        tts.speak(test_text)
        tts.wait_for_speech()
        
        time.sleep(1)
        
        # Test longer text with multiple sentences
        print("\n3. Testing longer text...")
        long_text = ("Systems check complete. All parameters are functioning within normal ranges. "
                    "The Hume AI voice synthesis is now operational and ready for deployment. "
                    "Shall I proceed with the integration into the main system?")
        print(f"   Speaking: '{long_text[:50]}...'")
        tts.speak(long_text)
        tts.wait_for_speech()
        
        time.sleep(1)
        
        # Test interruption capability
        print("\n4. Testing interruption (speaking for 3 seconds then stopping)...")
        interrupt_text = ("This is a very long message that will be interrupted. "
                         "I am going to keep talking for quite a while to demonstrate "
                         "the ability to stop playback mid-sentence. This feature is important "
                         "for responsive interaction when the user wants to interrupt.")
        print(f"   Speaking: '{interrupt_text[:50]}...'")
        tts.speak(interrupt_text)
        time.sleep(3)
        print("   Stopping playback...")
        tts.stop()
        tts.wait_for_speech()
        
        print("\n✅ All tests completed successfully!")
        print("\nThe Hume TTS integration is working correctly.")
        print("You can now use TTS_ENGINE='hume' in Jarvis.py")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()