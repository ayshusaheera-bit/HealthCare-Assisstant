import os
import numpy as np
import librosa
import noisereduce as nr
from scipy.io.wavfile import write


def clean_audio(input_file="audio/raw.wav", output_file="audio/cleaned.wav"):
    """
    Load an audio file, reduce noise, and save the cleaned audio.

    Args:
        input_file (str): Path to the input WAV file.
        output_file (str): Path to save the cleaned WAV file.

    Returns:
        str: Path to the cleaned audio file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load audio (resample to 16kHz)
    audio, sr = librosa.load(input_file, sr=16000)

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    # Convert to int16 for PCM WAV compatibility (required by speech_recognition)
    reduced_noise = (reduced_noise * 32767).astype(np.int16)

    # Save cleaned audio
    write(output_file, sr, reduced_noise)

    print("🧼 Cleaned audio saved:", output_file)
    return output_file


if __name__ == "__main__":
    clean_audio()
