import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from datetime import datetime
import re
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self):
        try:
            logger.info("Initializing EmotionAnalyzer with PyTorch backend...")

            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load model and tokenizer explicitly
            model_name = "SamLowe/roberta-base-go_emotions"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Move model to appropriate device
            self.model = self.model.to(self.device)

            # Initialize pipeline with explicit model and tokenizer
            self.classifier = pipeline(
                task="text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                device=self.device if self.device == "cuda" else -1
            )
            logger.info("EmotionAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing emotion classifier: {str(e)}")
            self.classifier = None

        # Enhanced emotion map with more nuanced emotions
        self.emotion_map = {
            'admiration': '🌟',
            'amusement': '😄',
            'anger': '😠',
            'annoyance': '😒',
            'approval': '👍',
            'caring': '🤗',
            'confusion': '😕',
            'curiosity': '🤔',
            'desire': '✨',
            'disappointment': '😞',
            'disapproval': '👎',
            'disgust': '🤢',
            'embarrassment': '😳',
            'excitement': '🎉',
            'fear': '😨',
            'gratitude': '🙏',
            'grief': '💔',
            'joy': '😊',
            'love': '❤️',
            'nervousness': '😰',
            'optimism': '🌈',
            'pride': '🦁',
            'realization': '💡',
            'relief': '😌',
            'remorse': '😔',
            'sadness': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }

    def preprocess_text(self, text):
        """Clean and prepare text for emotion analysis."""
        if not isinstance(text, str):
            logger.warning(f"Invalid input type: {type(text)}, expected str")
            return ""

        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove special characters but keep emoticons
            text = re.sub(r'[^\w\s!?.,\'"\(\)😀-🙏]', '', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return ""

    def analyze_text(self, text):
        """Analyze emotions in the given text with preprocessing."""
        if self.classifier is None:
            logger.error("Emotion classifier not initialized")
            return None

        try:
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text:
                logger.warning("Empty text after preprocessing")
                return None

            # Get emotion scores
            results = self.classifier(cleaned_text)[0]

            # Convert to DataFrame and sort by score
            emotions_df = pd.DataFrame(results)
            emotions_df = emotions_df.sort_values('score', ascending=False)

            # Filter out very low probability emotions (below 5%)
            emotions_df = emotions_df[emotions_df['score'] > 0.05]

            # Add metadata
            emotions_df['timestamp'] = datetime.now()
            emotions_df['text'] = cleaned_text

            logger.info(f"Successfully analyzed text. Found {len(emotions_df)} emotions above threshold")
            return emotions_df

        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return None

    def get_dominant_emotion(self, emotions_df):
        """Get the most prominent emotion from the analysis."""
        if emotions_df is None or emotions_df.empty:
            return 'neutral'
        return emotions_df.iloc[0]['label']

    def get_intensity(self, emotions_df):
        """Calculate the emotional intensity."""
        if emotions_df is None or emotions_df.empty:
            return 0
        return emotions_df.iloc[0]['score']

    def get_emotion_emoji(self, emotion):
        """Get the corresponding emoji for an emotion."""
        return self.emotion_map.get(emotion, '😐')