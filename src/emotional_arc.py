import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EmotionalArcAnalyzer:
    def __init__(self):
        self.emotion_weights = {
            'joy': 1.0,
            'love': 1.0,
            'excitement': 0.8,
            'optimism': 0.7,
            'pride': 0.6,
            'relief': 0.5,
            'neutral': 0.0,
            'confusion': -0.3,
            'nervousness': -0.4,
            'annoyance': -0.5,
            'sadness': -0.6,
            'disappointment': -0.7,
            'anger': -0.8,
            'fear': -0.9,
            'grief': -1.0
        }

    def calculate_emotional_intensity(self, emotions_df):
        """Calculate weighted emotional intensity based on emotion scores."""
        if emotions_df is None or emotions_df.empty:
            return 0.0

        total_intensity = 0.0
        total_weight = 0.0

        for _, row in emotions_df.iterrows():
            weight = self.emotion_weights.get(row['label'], 0.0)
            total_intensity += row['score'] * weight
            total_weight += abs(weight)

        return total_intensity / max(total_weight, 1.0)

    def analyze_emotional_arc(self, history_df, window_size=5):
        """Analyze emotional progression over time."""
        if history_df is None or history_df.empty:
            return pd.DataFrame()

        # Sort by timestamp
        history_df = history_df.sort_values('timestamp')

        # Calculate rolling statistics
        results = []
        window = []

        for _, row in history_df.iterrows():
            window.append({
                'timestamp': row['timestamp'],
                'intensity': self.calculate_emotional_intensity(pd.DataFrame([row]))
            })

            if len(window) > window_size:
                window.pop(0)

            # Calculate statistics for the current window
            intensities = [w['intensity'] for w in window]
            
            results.append({
                'timestamp': row['timestamp'],
                'current_intensity': intensities[-1],
                'trend': np.mean(intensities),
                'volatility': np.std(intensities),
                'peak': max(intensities),
                'valley': min(intensities)
            })

        return pd.DataFrame(results)

    def detect_emotional_shifts(self, arc_df, threshold=0.3):
        """Detect significant emotional shifts in the arc."""
        if arc_df.empty:
            return []

        shifts = []
        prev_intensity = arc_df.iloc[0]['current_intensity']

        for idx, row in arc_df.iterrows():
            intensity_change = row['current_intensity'] - prev_intensity
            if abs(intensity_change) >= threshold:
                shifts.append({
                    'timestamp': row['timestamp'],
                    'shift_magnitude': intensity_change,
                    'from_intensity': prev_intensity,
                    'to_intensity': row['current_intensity']
                })
            prev_intensity = row['current_intensity']

        return shifts
