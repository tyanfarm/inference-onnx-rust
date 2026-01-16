"""
Audio Preprocessing Script for ASR

This script converts audio files to 16kHz sample rate using librosa.
When audio is already at 16kHz, both Python and Rust ASR will produce identical results.

Usage:
    python preprocess_audio.py input.wav output_16k.wav
    python preprocess_audio.py audio_files/  # Convert all audio in directory
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install librosa soundfile numpy")
    sys.exit(1)


def convert_to_16k(input_path: str, output_path: str = None) -> str:
    """
    Convert audio file to 16kHz sample rate using librosa's default soxr_hq resampler.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file (optional, defaults to input_16k.wav)
    
    Returns:
        Path to the output file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_16k{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"Loading: {input_path}")
    
    # Load audio with librosa (this auto-resamples to sr if specified)
    audio, sr = librosa.load(str(input_path), sr=None)  # Load at native sample rate
    
    print(f"  Original: {sr} Hz, {len(audio)} samples, {len(audio)/sr:.2f}s")
    
    if sr != 16000:
        # Resample using librosa's default soxr_hq
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        print(f"  Resampled: 16000 Hz, {len(audio)} samples, {len(audio)/16000:.2f}s")
    else:
        print("  Already at 16kHz, no resampling needed")
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # Save as 16-bit WAV
    sf.write(str(output_path), audio, 16000, subtype='PCM_16')
    print(f"  Saved: {output_path}")
    
    return str(output_path)


def process_directory(dir_path: str, output_dir: str = None):
    """
    Convert all audio files in a directory to 16kHz.
    """
    dir_path = Path(dir_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = [f for f in dir_path.iterdir() 
                   if f.suffix.lower() in audio_extensions and '_16k' not in f.stem]
    
    if not audio_files:
        print(f"No audio files found in {dir_path}")
        return
    
    print(f"Found {len(audio_files)} audio files to convert\n")
    
    for audio_file in audio_files:
        if output_dir:
            output_path = output_dir / f"{audio_file.stem}_16k.wav"
        else:
            output_path = None
        
        try:
            convert_to_16k(str(audio_file), output_path)
        except Exception as e:
            print(f"  Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to 16kHz for ASR preprocessing"
    )
    parser.add_argument(
        "input",
        help="Input audio file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory",
        default=None
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        process_directory(str(input_path), args.output)
    elif input_path.is_file():
        convert_to_16k(str(input_path), args.output)
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
