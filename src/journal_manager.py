import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class JournalEntry:
    timestamp: datetime
    text: str
    emotions: Dict[str, float]
    context: str
    tags: List[str]

class JournalManager:
    def __init__(self):
        self.entries = pd.DataFrame(columns=[
            'timestamp', 'text', 'emotions', 'context', 'tags',
            'primary_emotion', 'emotional_intensity'
        ])
        
    def add_entry(self, text: str, emotions_df: pd.DataFrame, context: str = "", tags: List[str] = None) -> bool:
        """Add a new journal entry with emotional analysis."""
        try:
            if emotions_df is None or emotions_df.empty:
                logger.warning("No emotions detected for journal entry")
                return False
                
            # Convert emotions to dictionary format
            emotions_dict = dict(zip(emotions_df['label'], emotions_df['score']))
            
            # Get primary emotion and intensity
            primary_emotion = emotions_df.iloc[0]['label']
            emotional_intensity = emotions_df.iloc[0]['score']
            
            # Create new entry
            new_entry = pd.DataFrame([{
                'timestamp': datetime.now(),
                'text': text,
                'emotions': emotions_dict,
                'context': context,
                'tags': tags or [],
                'primary_emotion': primary_emotion,
                'emotional_intensity': emotional_intensity
            }])
            
            # Drop empty or all-NA columns before concatenating
            new_entry = new_entry.dropna(how='all', axis=1)
            
            # Add to entries
            self.entries = pd.concat([self.entries, new_entry], ignore_index=True)
            logger.info("Successfully added new journal entry")
            return True
            
        except Exception as e:
            logger.error(f"Error adding journal entry: {str(e)}")
            return False
            
    def get_entries(self, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None,
                   emotion_filter: Optional[str] = None) -> pd.DataFrame:
        """Retrieve journal entries with optional filtering."""
        if self.entries.empty:
            return pd.DataFrame()
            
        filtered_entries = self.entries.copy()
        
        if start_date:
            filtered_entries = filtered_entries[filtered_entries['timestamp'] >= start_date]
        if end_date:
            filtered_entries = filtered_entries[filtered_entries['timestamp'] <= end_date]
        if emotion_filter:
            filtered_entries = filtered_entries[filtered_entries['primary_emotion'] == emotion_filter]
            
        return filtered_entries.sort_values('timestamp', ascending=False)
        
    def generate_insights(self, timeframe_days: int = 7) -> Dict:
        """Generate insights from journal entries."""
        if self.entries.empty:
            return {
                'summary': "No journal entries available for analysis.",
                'patterns': [],
                'recommendations': []
            }

        # Calculate the start date for filtering
        start_date = datetime.now() - timedelta(days=timeframe_days)

        # Filter entries within the timeframe
        recent_entries = self.entries[self.entries['timestamp'] >= start_date]

        if recent_entries.empty:
            return {
                'summary': f"No journal entries in the last {timeframe_days} days.",
                'patterns': [],
                'recommendations': []
            }

        # Analyze emotional patterns
        emotion_counts = pd.DataFrame([
            emotion for emotions in recent_entries['emotions'] for emotion in emotions.items()
        ], columns=['emotion', 'score']).groupby('emotion')['score'].mean()

        dominant_emotions = emotion_counts.nlargest(3)

        # Generate insights
        insights = {
            'summary': f"Analysis based on {len(recent_entries)} entries over {timeframe_days} days",
            'patterns': [
                f"Your dominant emotion was {emotion} (intensity: {score:.2f})"
                for emotion, score in dominant_emotions.items()
            ],
            'recommendations': self._generate_recommendations(dominant_emotions)
        }

        return insights
        
    def _generate_recommendations(self, dominant_emotions: pd.Series) -> List[str]:
        """Generate personalized recommendations based on emotional patterns."""
        recommendations = []
        
        # Example recommendation logic
        for emotion, score in dominant_emotions.items():
            if emotion in ['joy', 'excitement', 'optimism']:
                recommendations.append(
                    f"Your high levels of {emotion} suggest a positive period. "
                    "Consider journaling about what's working well to maintain this momentum."
                )
            elif emotion in ['sadness', 'anger', 'fear']:
                recommendations.append(
                    f"You've been experiencing {emotion} recently. "
                    "Try incorporating mindfulness or stress-relief activities into your routine."
                )
            elif emotion in ['neutral', 'confusion']:
                recommendations.append(
                    f"Your {emotion} entries indicate a period of transition. "
                    "This might be a good time for reflection and goal-setting."
                )
                
        return recommendations
    
    def get_emotion_trends(self) -> pd.DataFrame:
        """Calculate emotion trends over time."""
        if self.entries.empty:
            return pd.DataFrame()
            
        # Unpack emotions into separate rows
        trends = []
        for _, entry in self.entries.iterrows():
            for emotion, score in entry['emotions'].items():
                trends.append({
                    'timestamp': entry['timestamp'],
                    'emotion': emotion,
                    'score': score
                })
                
        return pd.DataFrame(trends)