#!/usr/bin/env python3
"""
Test script for Mac-optimized Chatterbox TTS
Verifies MPS acceleration and voice synthesis
"""
import torch
import torchaudio as ta
import os
import time

def test_mac_tts():
    print("🧪 Testing Chatterbox TTS on Mac")
    print("=" * 50)
    
    # Check device availability
    print("\n1️⃣ Checking device support...")
    if torch.backends.mps.is_available():
        device = "mps"
        print("   ✅ Apple Metal Performance Shaders (MPS) available!")
        print("   🚀 Using hardware acceleration")
    else:
        device = "cpu"
        print("   ℹ️  MPS not available, using CPU")
    
    # Setup device mapping for model loading
    print("\n2️⃣ Setting up device mapping...")
    map_location = torch.device(device)
    
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    
    torch.load = patched_torch_load
    print("   ✅ Device mapping configured")
    
    # Load model
    print("\n3️⃣ Loading Chatterbox model...")
    start_time = time.time()
    
    try:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device)
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f} seconds")
        print(f"   📍 Running on: {device}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Check for JARVIS voice
    print("\n4️⃣ Checking for JARVIS voice sample...")
    voice_paths = [
        "audio_library/jarvis_sample.wav",
        "audio_library/jarvis_sample.aif",
        "jarvis_sample.wav",
        "jarvis_sample.aif"
    ]
    
    jarvis_voice = None
    for path in voice_paths:
        if os.path.exists(path):
            jarvis_voice = path
            print(f"   ✅ Found JARVIS voice at: {path}")
            break
    else:
        print("   ⚠️  JARVIS voice not found, using default voice")
    
    # Test generation
    print("\n5️⃣ Testing speech synthesis...")
    test_text = "Good evening, Sir. All systems are now online. The Mac optimization is working perfectly."
    
    print(f"   Text: '{test_text}'")
    print("   Generating audio...")
    
    start_time = time.time()
    
    try:
        if jarvis_voice:
            wav = model.generate(
                test_text,
                audio_prompt_path=jarvis_voice,
                exaggeration=0.3,
                cfg_weight=0.5
            )
        else:
            wav = model.generate(
                test_text,
                exaggeration=0.3,
                cfg_weight=0.5
            )
        
        gen_time = time.time() - start_time
        
        # Save the output
        output_file = "test_mac_output.wav"
        ta.save(output_file, wav, model.sr)
        
        print(f"   ✅ Audio generated in {gen_time:.2f} seconds")
        print(f"   💾 Saved to: {output_file}")
        print(f"   🎵 Sample rate: {model.sr} Hz")
        print(f"   📏 Duration: {wav.shape[1] / model.sr:.2f} seconds")
        
        # Performance metrics
        rtf = gen_time / (wav.shape[1] / model.sr)
        print(f"   ⚡ Real-time factor: {rtf:.2f}x (lower is better)")
        
        if rtf < 1.0:
            print("   🎉 Faster than real-time!")
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return
    
    # Test streaming if available
    print("\n6️⃣ Testing streaming capability...")
    if hasattr(model, 'generate_stream'):
        print("   ✅ Streaming generation available!")
        
        try:
            chunks = 0
            first_chunk_time = None
            
            for audio_chunk, metrics in model.generate_stream(
                "Testing streaming synthesis.",
                exaggeration=0.3,
                cfg_weight=0.5
            ):
                chunks += 1
                if chunks == 1 and hasattr(metrics, 'latency_to_first_chunk'):
                    first_chunk_time = metrics.latency_to_first_chunk
            
            print(f"   📦 Generated {chunks} chunks")
            if first_chunk_time:
                print(f"   ⏱️  First chunk latency: {first_chunk_time:.3f}s")
        except Exception as e:
            print(f"   ⚠️  Streaming test failed: {e}")
    else:
        print("   ℹ️  Streaming not available in this version")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ TTS Test Complete!")
    print("\nRecommendations:")
    
    if device == "mps":
        print("• You're using Metal acceleration - optimal for Mac! 🍎")
    else:
        print("• Consider upgrading to Apple Silicon for better performance")
    
    if jarvis_voice:
        print("• JARVIS voice is configured correctly")
    else:
        print("• Add a JARVIS voice sample for authentic speech")
    
    if rtf < 0.5:
        print("• Excellent performance - very fast generation!")
    elif rtf < 1.0:
        print("• Good performance - faster than real-time")
    else:
        print("• Consider reducing quality settings for faster generation")
    
    print("\nTo play the test audio:")
    print(f"  afplay {output_file}")

if __name__ == "__main__":
    test_mac_tts()