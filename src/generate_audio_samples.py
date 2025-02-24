import numpy as np
from scipy.io import wavfile
import os

def generate_audio_samples():
    """Generate sample audio files for different moods"""
    # Create assets directory if it doesn't exist
    os.makedirs('assets/audio', exist_ok=True)
    
    # Sample rate for audio files
    sample_rate = 44100
    duration = 3  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Generate different tones for different moods
    audio_files = {
        'anxious': 0.5 * np.sin(2 * np.pi * 440 * t),  # 440 Hz - A4 note
        'stressed': 0.5 * np.sin(2 * np.pi * 392 * t),  # 392 Hz - G4 note
        'sad': 0.5 * np.sin(2 * np.pi * 349.23 * t),   # 349.23 Hz - F4 note
        'happy': 0.5 * np.sin(2 * np.pi * 523.25 * t), # 523.25 Hz - C5 note
        'neutral': 0.5 * np.sin(2 * np.pi * 261.63 * t) # 261.63 Hz - C4 note
    }
    
    # Save each audio sample
    for mood, audio_data in audio_files.items():
        filename = f'assets/audio/{mood}.wav'
        wavfile.write(filename, sample_rate, audio_data.astype(np.float32))

if __name__ == '__main__':
    generate_audio_samples()
