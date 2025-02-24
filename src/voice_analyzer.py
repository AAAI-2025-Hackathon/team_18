import sounddevice as sd
import numpy as np
import librosa
from scipy.io.wavfile import write
import tempfile
import os
from datetime import datetime

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate for speech recognition
        self.duration = 5  # Recording duration in seconds
        
    def record_audio(self):
        """Record audio from microphone"""
        print("Recording...")
        audio_data = sd.rec(
            int(self.sample_rate * self.duration),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()  # Wait until recording is done
        return audio_data
        
    def save_audio(self, audio_data):
        """Save audio to a temporary WAV file"""
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(temp_dir, f"audio_{timestamp}.wav")
        write(filename, self.sample_rate, audio_data)
        return filename
        
    def extract_features(self, audio_data):
        """Extract audio features for emotion analysis"""
        # Extract various audio features
        mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=self.sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_data.flatten(), sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_data.flatten(), sr=self.sample_rate)
        
        # Calculate statistics for each feature
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'mel_mean': np.mean(mel, axis=1)
        }
        
        return features
        
    def analyze_emotion(self, audio_data):
        """Analyze emotion from audio features"""
        features = self.extract_features(audio_data)
        
        # Placeholder for emotion classification
        # In a real implementation, we would use a trained model here
        # For now, return a simplified analysis based on audio energy
        energy = np.mean(np.abs(audio_data))
        
        # Simple rule-based classification for demonstration
        if energy > 0.1:
            return {'label': 'excited', 'score': min(energy, 1.0)}
        elif energy > 0.05:
            return {'label': 'neutral', 'score': 0.5}
        else:
            return {'label': 'calm', 'score': max(1 - energy, 0.0)}
