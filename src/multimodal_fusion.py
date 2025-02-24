import numpy as np
import pandas as pd
from datetime import datetime

class MultiModalFusion:
    def __init__(self):
        self.modality_weights = {
            'text': 0.4,
            'voice': 0.3,
            'facial': 0.3
        }
    
    def normalize_emotion_labels(self, emotion_dict):
        """Normalize emotion labels across different modalities."""
        # Common emotion mapping
        emotion_map = {
            # Text emotions
            'joy': 'happy',
            'excitement': 'happy',
            'love': 'happy',
            'optimism': 'happy',
            'admiration': 'happy',
            'amusement': 'happy',
            
            'sadness': 'sad',
            'disappointment': 'sad',
            'grief': 'sad',
            
            'anger': 'angry',
            'annoyance': 'angry',
            'disapproval': 'angry',
            
            'fear': 'fearful',
            'nervousness': 'fearful',
            
            'neutral': 'neutral',
            
            # Voice/Facial specific
            'calm': 'neutral',
            'excited': 'happy',
            'attentive': 'neutral'
        }
        
        normalized = {}
        for emotion, score in emotion_dict.items():
            mapped_emotion = emotion_map.get(emotion.lower(), 'neutral')
            normalized[mapped_emotion] = normalized.get(mapped_emotion, 0) + score
            
        # Normalize scores
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v/total for k, v in normalized.items()}
            
        return normalized
    
    def fuse_emotions(self, text_result=None, voice_result=None, facial_result=None):
        """Fuse emotions from different modalities."""
        try:
            results = []
            timestamp = datetime.now()
            
            # Process text analysis
            if text_result is not None and not text_result.empty:
                text_emotions = dict(zip(text_result['label'], text_result['score']))
                normalized_text = self.normalize_emotion_labels(text_emotions)
                results.append({
                    'modality': 'text',
                    'emotions': normalized_text,
                    'weight': self.modality_weights['text']
                })
            
            # Process voice analysis
            if voice_result is not None:
                voice_emotions = {voice_result['label']: voice_result['score']}
                normalized_voice = self.normalize_emotion_labels(voice_emotions)
                results.append({
                    'modality': 'voice',
                    'emotions': normalized_voice,
                    'weight': self.modality_weights['voice']
                })
            
            # Process facial analysis
            if facial_result is not None:
                facial_emotions = {facial_result['label']: facial_result['score']}
                normalized_facial = self.normalize_emotion_labels(facial_emotions)
                results.append({
                    'modality': 'facial',
                    'emotions': normalized_facial,
                    'weight': self.modality_weights['facial']
                })
            
            if not results:
                return None
            
            # Combine emotions with weights
            combined_emotions = {}
            for result in results:
                for emotion, score in result['emotions'].items():
                    weighted_score = score * result['weight']
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + weighted_score
            
            # Create final DataFrame
            emotions_df = pd.DataFrame([
                {'label': emotion, 'score': score, 'timestamp': timestamp}
                for emotion, score in combined_emotions.items()
            ])
            
            return emotions_df.sort_values('score', ascending=False)
            
        except Exception as e:
            print(f"Error in emotion fusion: {str(e)}")
            return None
    
    def get_confidence_scores(self, text_result=None, voice_result=None, facial_result=None):
        """Calculate confidence scores for each modality."""
        confidence = {}
        
        if text_result is not None and not text_result.empty:
            confidence['text'] = text_result['score'].max()
            
        if voice_result is not None:
            confidence['voice'] = voice_result['score']
            
        if facial_result is not None:
            confidence['facial'] = facial_result['score']
            
        return confidence
