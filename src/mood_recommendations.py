from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import os

@dataclass
class AudioRecommendation:
    title: str
    category: str  # 'music' or 'meditation'
    mood_tags: List[str]
    duration: int  # in minutes
    intensity: str  # 'low', 'medium', 'high'
    description: str
    recommended_time: str  # e.g., 'morning', 'evening'
    audio_file: str = ""  # Path to audio file

class MoodRecommendationManager:
    def __init__(self):
        # Audio files paths
        self.audio_files = {
            'anxious': 'assets/audio/anxious.wav',
            'stressed': 'assets/audio/stressed.wav',
            'sad': 'assets/audio/sad.wav',
            'happy': 'assets/audio/happy.wav',
            'neutral': 'assets/audio/neutral.wav'
        }

        # Initialize meditation recommendations database
        self.meditation_database = {
            'anxious': [
                AudioRecommendation(
                    title="Calming Breath Work",
                    category="meditation",
                    mood_tags=["anxiety", "stress-relief"],
                    duration=10,
                    intensity="low",
                    description="Gentle breathing exercises to reduce anxiety",
                    recommended_time="morning",
                    audio_file=self.audio_files['anxious']
                ),
                AudioRecommendation(
                    title="Body Scan Relaxation",
                    category="meditation",
                    mood_tags=["anxiety", "relaxation"],
                    duration=15,
                    intensity="low",
                    description="Progressive body scan for deep relaxation",
                    recommended_time="evening",
                    audio_file=self.audio_files['anxious']
                )
            ],
            'stressed': [
                AudioRecommendation(
                    title="Stress Release Meditation",
                    category="meditation",
                    mood_tags=["stress-relief", "relaxation"],
                    duration=20,
                    intensity="medium",
                    description="Guided meditation for stress relief",
                    recommended_time="afternoon",
                    audio_file=self.audio_files['stressed']
                )
            ],
            'sad': [
                AudioRecommendation(
                    title="Uplifting Mindfulness",
                    category="meditation",
                    mood_tags=["mood-boost", "mindfulness"],
                    duration=15,
                    intensity="medium",
                    description="Mindfulness practice to lift your spirits",
                    recommended_time="morning",
                    audio_file=self.audio_files['sad']
                )
            ],
            'happy': [
                AudioRecommendation(
                    title="Gratitude Meditation",
                    category="meditation",
                    mood_tags=["gratitude", "joy"],
                    duration=10,
                    intensity="medium",
                    description="Practice gratitude and joy",
                    recommended_time="morning",
                    audio_file=self.audio_files['happy']
                )
            ],
            'neutral': [
                AudioRecommendation(
                    title="Mindful Awareness",
                    category="meditation",
                    mood_tags=["mindfulness", "balance"],
                    duration=15,
                    intensity="medium",
                    description="General mindfulness meditation for any time",
                    recommended_time="any",
                    audio_file=self.audio_files['neutral']
                ),
                AudioRecommendation(
                    title="Breathing Focus",
                    category="meditation",
                    mood_tags=["focus", "presence"],
                    duration=10,
                    intensity="low",
                    description="Simple breathing meditation",
                    recommended_time="any",
                    audio_file=self.audio_files['neutral']
                )
            ]
        }

        # Initialize music recommendations database with similar structure
        self.music_database = {
            'anxious': [
                AudioRecommendation(
                    title="Calming Classical",
                    category="music",
                    mood_tags=["calm", "soothing"],
                    duration=30,
                    intensity="low",
                    description="Soft classical pieces for anxiety relief",
                    recommended_time="any",
                    audio_file=self.audio_files['anxious']
                )
            ],
            'stressed': [
                AudioRecommendation(
                    title="Ambient Relaxation",
                    category="music",
                    mood_tags=["ambient", "relaxation"],
                    duration=40,
                    intensity="low",
                    description="Ambient music for stress relief",
                    recommended_time="evening",
                    audio_file=self.audio_files['stressed']
                )
            ],
            'sad': [
                AudioRecommendation(
                    title="Upbeat Instrumental",
                    category="music",
                    mood_tags=["uplifting", "energetic"],
                    duration=35,
                    intensity="medium",
                    description="Uplifting instrumental tracks",
                    recommended_time="morning",
                    audio_file=self.audio_files['sad']
                )
            ],
            'happy': [
                AudioRecommendation(
                    title="Feel-Good Playlist",
                    category="music",
                    mood_tags=["joyful", "energetic"],
                    duration=45,
                    intensity="high",
                    description="Upbeat and energetic music",
                    recommended_time="any",
                    audio_file=self.audio_files['happy']
                )
            ],
            'neutral': [
                AudioRecommendation(
                    title="Balanced Mix",
                    category="music",
                    mood_tags=["balanced", "focus"],
                    duration=40,
                    intensity="medium",
                    description="Well-balanced instrumental music",
                    recommended_time="any",
                    audio_file=self.audio_files['neutral']
                )
            ]
        }

    def get_recommendations(self, 
                          emotional_state: str, 
                          time_of_day: Optional[str] = None,
                          duration_preference: Optional[int] = None) -> Dict[str, List[AudioRecommendation]]:
        """Get personalized audio recommendations based on emotional state and preferences"""
        emotional_state = emotional_state.lower()
        if emotional_state not in self.meditation_database:
            emotional_state = 'neutral'

        recommendations = {
            'meditation': [],
            'music': []
        }

        # Get meditation recommendations
        meditations = self.meditation_database.get(emotional_state, [])
        if time_of_day:
            meditations = [m for m in meditations if m.recommended_time in [time_of_day, 'any']]
        if duration_preference:
            meditations = [m for m in meditations if m.duration <= duration_preference]
        recommendations['meditation'] = meditations

        # Get music recommendations
        music = self.music_database.get(emotional_state, [])
        if time_of_day:
            music = [m for m in music if m.recommended_time in [time_of_day, 'any']]
        if duration_preference:
            music = [m for m in music if m.duration <= duration_preference]
        recommendations['music'] = music

        return recommendations

    def get_time_of_day(self) -> str:
        """Determine the time of day for recommendations"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        else:
            return 'evening'

    def format_duration(self, minutes: int) -> str:
        """Format duration in a human-readable way"""
        if minutes < 60:
            return f"{minutes} minutes"
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minutes"