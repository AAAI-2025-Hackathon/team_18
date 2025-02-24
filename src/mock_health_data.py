from datetime import datetime, timedelta
import random
from typing import Dict, List
import numpy as np

class HealthDataSimulator:
    def __init__(self):
        self.base_heart_rate = random.randint(60, 75)
        self.base_stress_level = random.randint(30, 50)

    def generate_daily_data(self) -> Dict[str, float]:
        """Generate mock health data for a single day"""
        # Add some random variation to base values
        heart_rate = max(45, min(120, self.base_heart_rate + random.randint(-5, 5)))
        stress_level = max(0, min(100, self.base_stress_level + random.randint(-10, 10)))

        # Create the data dictionary with proper timestamp type
        data = {
            'heart_rate': float(heart_rate),
            'sleep_score': float(random.randint(60, 95)),
            'readiness_score': float(random.randint(50, 100)),
            'activity_score': float(random.randint(40, 100)),
            'stress_level': float(stress_level),
            'timestamp': datetime.now()  # Keep timestamp as datetime
        }
        return data

    def generate_weekly_data(self) -> List[Dict[str, float]]:
        """Generate a week's worth of mock health data"""
        weekly_data = []
        for i in range(7):
            day_data = self.generate_daily_data()
            # Properly set timestamp for each day
            day_data['timestamp'] = datetime.now() - timedelta(days=6-i)
            weekly_data.append(day_data)
        return weekly_data

    def generate_trends(self) -> Dict[str, List[float]]:
        """Generate trending data for visualization"""
        days = 7
        base_values = {
            'heart_rate': self.base_heart_rate,
            'stress_level': self.base_stress_level,
            'sleep_quality': 75,
            'activity_level': 70
        }

        trends = {}
        for metric, base in base_values.items():
            # Generate smooth random walk
            noise = np.random.normal(0, 2, days)
            trend = np.cumsum(noise) + base
            # Normalize to reasonable ranges
            trend = np.clip(trend, 0, 100)
            trends[metric] = trend.tolist()

        return trends