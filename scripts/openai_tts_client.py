"""
OpenAI API Client for Kokoros TTS Server

This script demonstrates how to use the OpenAI Python SDK to interact with
the Kokoros TTS server, which provides an OpenAI-compatible API.

Prerequisites:
1. Install dependencies: pip install openai
2. Start the Kokoros server: ./target/release/koko openai
   (Server runs on http://localhost:3000 by default)

Usage:
    python openai_tts_client.py
"""

from openai import OpenAI
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:3000/v1"
API_KEY = "any-key-works"  # The Kokoros server doesn't require a real key

# Available Kokoro voices (see README for full list)
# Female voices: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, 
#                af_nicole, af_nova, af_river, af_sarah, af_sky
# Male voices: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx
# British voices: bf_alice, bf_emma, bf_isabella, bf_lily, bm_george, bm_lewis, bm_oliver

# Initialize the OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def generate_speech_simple(text: str, voice: str = "af_sky", output_path: str = "output.wav"):
    """
    Generate speech from text using the Kokoros TTS server.
    
    Args:
        text: The text to convert to speech
        voice: The voice to use (e.g., 'af_sky', 'am_michael')
        output_path: Path to save the generated audio file
    """
    response = client.audio.speech.create(
        model="tts-1",  # Model name is ignored by Kokoros, but required by OpenAI SDK
        voice=voice,
        input=text,
    )
    response.write_to_file(output_path)
    print(f"✓ Audio saved to: {output_path}")


def generate_speech_with_options(
    text: str,
    voice: str = "af_sky",
    speed: float = 1.0,
    response_format: str = "wav",  # wav, mp3, opus, pcm
    output_path: str = "output_advanced.wav"
):
    """
    Generate speech with additional options.
    
    Args:
        text: The text to convert to speech
        voice: The voice to use
        speed: Playback speed (0.25 to 4.0)
        response_format: Audio format (wav, mp3, opus, pcm)
        output_path: Path to save the generated audio file
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed,
        response_format=response_format,
    )
    response.write_to_file(output_path)
    print(f"✓ Audio saved to: {output_path}")


def generate_speech_streaming(text: str, voice: str = "af_sky", output_path: str = "streaming_output.pcm"):
    """
    Generate speech with streaming (returns audio chunks as they're generated).
    Note: Streaming only supports PCM format.
    
    Args:
        text: The text to convert to speech
        voice: The voice to use
        output_path: Path to save the generated audio file
    """
    import requests
    
    response = requests.post(
        f"{BASE_URL[:-3]}/v1/audio/speech",  # Remove /v1 from base URL
        headers={"Content-Type": "application/json"},
        json={
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "stream": True
        },
        stream=True
    )
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print(f"✓ Streaming audio saved to: {output_path}")
    print("  Note: PCM format - sample rate: 24000 Hz, 16-bit signed little-endian mono")


def generate_speech_with_style_mixing(text: str, output_path: str = "style_mixed.wav"):
    """
    Generate speech using style mixing (combining multiple voices).
    Format: "voice1.weight+voice2.weight" (weights should sum to ~1.0)
    
    Example: "af_sky.4+af_nicole.5" mixes 40% af_sky with 50% af_nicole
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="af_sky.4+af_nicole.5",  # Style mixing syntax
        input=text,
    )
    response.write_to_file(output_path)
    print(f"✓ Style-mixed audio saved to: {output_path}")


def list_available_voices():
    """Print all available Kokoro voices."""
    voices = {
        "American Female": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", 
                           "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
        "American Male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                         "am_michael", "am_onyx"],
        "British Female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
        "British Male": ["bm_george", "bm_lewis", "bm_oliver"],
    }
    
    print("\n=== Available Kokoro Voices ===")
    for category, voice_list in voices.items():
        print(f"\n{category}:")
        for voice in voice_list:
            print(f"  - {voice}")


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Simple text-to-speech
    print("\n--- Example 1: Simple TTS ---")
    generate_speech_simple(
        text="Hello! Welcome to Kokoros, a fast text-to-speech engine powered by Rust.",
        voice="af_sky",
        output_path=str(output_dir / "example1_simple.wav")
    )
    
    # Example 2: Different voice with speed adjustment
    print("\n--- Example 2: Different Voice with Speed ---")
    generate_speech_with_options(
        text="This is a demonstration using a male voice at a slightly faster speed.",
        voice="am_michael",
        speed=1.2,
        response_format="wav",
        output_path=str(output_dir / "example2_male_voice.wav")
    )
    
    # Example 3: MP3 output format
    print("\n--- Example 3: MP3 Format ---")
    generate_speech_with_options(
        text="This audio is saved in MP3 format for better compression.",
        voice="af_bella",
        response_format="mp3",
        output_path=str(output_dir / "example3_mp3_format.mp3")
    )
    
    # Example 4: Style mixing (ASMR-like effect)
    print("\n--- Example 4: Style Mixing ---")
    generate_speech_with_style_mixing(
        text="This uses style mixing to create a unique voice blend.",
        output_path=str(output_dir / "example4_style_mixed.wav")
    )
    
    # Example 5: Long text
    print("\n--- Example 5: Long Text ---")
    long_text = """
    Welcome to our comprehensive technology demonstration session. 
    Today we will explore advanced parallel processing systems thoroughly. 
    These systems utilize multiple computational instances simultaneously for efficiency.
    Quality assurance validates each processing stage thoroughly before deployment.
    """
    generate_speech_simple(
        text=long_text,
        voice="af_nova",
        output_path=str(output_dir / "example5_long_text.wav")
    )
    
    # List all available voices
    list_available_voices()
    
    print("\n" + "="*50)
    print("All examples completed! Check the 'tmp/' directory for audio files.")
    print("="*50)
