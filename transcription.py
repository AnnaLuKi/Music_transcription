import sys
import numpy as np
import librosa

def transcribe(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)

    f0, voiced_flag, confidence = librosa.pyin(
        audio,
        fmin=100,
        fmax=1500,
        sr=sr
    )

    for freq, voiced, conf in zip(f0, voiced_flag, confidence):
        if voiced and not np.isnan(freq):
            note = librosa.hz_to_note(freq)
            print(f"{note}: {freq:.2f} Hz (confidence: {conf:.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py recording.wav")
        sys.exit(1)

    transcribe(sys.argv[1])
