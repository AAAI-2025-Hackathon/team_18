from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, time
import json

@dataclass
class HealthProfile:
    # Physical metrics
    height: float
    weight: float
    age: int
    gender: str
    activity_level: str  # sedentary, light, moderate, very_active

    # Health goals
    primary_goal: str  # weight_loss, muscle_gain, maintenance, stress_reduction
    target_weight: Optional[float] = None

    # Preferences
    dietary_restrictions: List[str] = None  # Initialize with empty list
    preferred_exercise_types: List[str] = None  # Initialize with empty list
    wake_time: time = time(6, 0)  # Default 6:00 AM
    sleep_time: time = time(22, 0)  # Default 10:00 PM

    def __post_init__(self):
        # Initialize lists if None
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []
        if self.preferred_exercise_types is None:
            self.preferred_exercise_types = []

@dataclass
class HealthMetrics:
    heart_rate: float
    sleep_score: float
    readiness_score: float
    activity_score: float
    stress_level: float
    emotional_state: str
    emotional_intensity: float
    timestamp: datetime

class HealthPlanGenerator:
    def __init__(self):
        self.exercise_intensity_map = {
            'sedentary': 'light',
            'light': 'moderate',
            'moderate': 'moderate to high',
            'very_active': 'high'
        }
        
        self.emotion_exercise_map = {
            'stressed': ['yoga', 'walking', 'stretching'],
            'anxious': ['meditation', 'light cardio', 'swimming'],
            'energetic': ['hiit', 'running', 'strength training'],
            'tired': ['gentle stretching', 'walking', 'light yoga'],
            'neutral': ['mixed cardio', 'bodyweight exercises', 'cycling']
        }
        
        self.stress_relief_activities = {
            'high': ['deep breathing', 'meditation', 'progressive muscle relaxation'],
            'moderate': ['light walking', 'gentle stretching', 'mindfulness'],
            'low': ['regular exercise', 'social activities', 'hobby engagement']
        }
    
    def generate_plans(self, profile: HealthProfile, metrics: HealthMetrics) -> Dict:
        """Generate comprehensive health plans based on user profile and metrics"""
        plans = {
            'exercise': self._create_exercise_plan(profile, metrics),
            'diet': self._create_diet_plan(profile, metrics),
            'sleep': self._create_sleep_plan(profile, metrics),
            'stress_management': self._create_stress_management_plan(metrics)
        }
        return plans

    def _create_exercise_plan(self, profile: HealthProfile, metrics: HealthMetrics) -> Dict:
        """Generate personalized exercise plan"""
        base_intensity = self.exercise_intensity_map[profile.activity_level]
        
        # Adjust based on readiness and emotional state
        if metrics.readiness_score < 60:
            intensity_modifier = "lower"
        elif metrics.readiness_score > 80:
            intensity_modifier = "higher"
        else:
            intensity_modifier = "maintain"
            
        # Get emotion-based exercises
        recommended_exercises = self.emotion_exercise_map.get(
            metrics.emotional_state.lower(), 
            self.emotion_exercise_map['neutral']
        )
        
        # Create weekly schedule
        weekly_schedule = {
            'Monday': {
                'focus': 'Cardio',
                'intensity': base_intensity,
                'suggested_activities': recommended_exercises[:2],
                'duration': '30-45 minutes'
            },
            'Tuesday': {
                'focus': 'Strength',
                'intensity': base_intensity,
                'suggested_activities': ['bodyweight exercises', 'resistance training'],
                'duration': '40-50 minutes'
            },
            'Wednesday': {
                'focus': 'Recovery',
                'intensity': 'light',
                'suggested_activities': ['walking', 'stretching'],
                'duration': '20-30 minutes'
            },
            'Thursday': {
                'focus': 'Mixed',
                'intensity': base_intensity,
                'suggested_activities': recommended_exercises,
                'duration': '30-45 minutes'
            },
            'Friday': {
                'focus': 'Strength',
                'intensity': base_intensity,
                'suggested_activities': ['resistance training', 'core exercises'],
                'duration': '40-50 minutes'
            },
            'Saturday': {
                'focus': 'Active Recovery',
                'intensity': 'moderate',
                'suggested_activities': ['yoga', 'swimming'],
                'duration': '30-40 minutes'
            },
            'Sunday': {
                'focus': 'Rest',
                'intensity': 'very light',
                'suggested_activities': ['walking', 'gentle stretching'],
                'duration': '15-20 minutes'
            }
        }
        
        return {
            'weekly_schedule': weekly_schedule,
            'intensity_level': base_intensity,
            'intensity_modifier': intensity_modifier,
            'recommended_exercises': recommended_exercises,
            'notes': self._generate_exercise_notes(profile, metrics)
        }

    def _create_diet_plan(self, profile: HealthProfile, metrics: HealthMetrics) -> Dict:
        """Generate personalized diet plan"""
        # Base recommendations
        base_meals = {
            'breakfast': {
                'timing': '7:00 AM',
                'components': ['protein', 'complex carbs', 'fruits'],
                'suggestions': [
                    'Oatmeal with berries and nuts',
                    'Greek yogurt parfait',
                    'Whole grain toast with eggs'
                ]
            },
            'lunch': {
                'timing': '12:30 PM',
                'components': ['lean protein', 'vegetables', 'whole grains'],
                'suggestions': [
                    'Grilled chicken salad',
                    'Quinoa bowl with vegetables',
                    'Turkey wrap with avocado'
                ]
            },
            'dinner': {
                'timing': '6:30 PM',
                'components': ['protein', 'vegetables', 'healthy fats'],
                'suggestions': [
                    'Baked fish with roasted vegetables',
                    'Stir-fry with tofu',
                    'Lean meat with sweet potato'
                ]
            },
            'snacks': {
                'timing': ['10:00 AM', '3:30 PM'],
                'suggestions': [
                    'Apple with almond butter',
                    'Carrot sticks with hummus',
                    'Mixed nuts and dried fruit'
                ]
            }
        }
        
        # Adjust based on stress and emotion
        if metrics.stress_level > 70:
            stress_foods = [
                'Dark chocolate',
                'Green tea',
                'Nuts and seeds',
                'Berries',
                'Leafy greens'
            ]
        else:
            stress_foods = []
            
        return {
            'meal_schedule': base_meals,
            'stress_reducing_foods': stress_foods,
            'hydration': self._calculate_hydration_needs(profile, metrics),
            'notes': self._generate_diet_notes(profile, metrics)
        }

    def _create_sleep_plan(self, profile: HealthProfile, metrics: HealthMetrics) -> Dict:
        """Generate personalized sleep plan"""
        # Calculate optimal sleep window
        optimal_duration = self._calculate_sleep_duration(profile, metrics)
        
        # Create sleep schedule
        sleep_schedule = {
            'target_bedtime': profile.sleep_time.strftime('%I:%M %p'),
            'target_wake_time': profile.wake_time.strftime('%I:%M %p'),
            'optimal_duration': optimal_duration
        }
        
        # Generate wind-down routine
        wind_down_routine = [
            {'time': '1 hour before bed', 'activity': 'Dim lights and avoid screens'},
            {'time': '45 minutes before bed', 'activity': 'Light stretching or reading'},
            {'time': '30 minutes before bed', 'activity': 'Relaxation exercises'},
            {'time': '15 minutes before bed', 'activity': 'Meditation or deep breathing'}
        ]
        
        # Adjust based on stress and emotion
        if metrics.stress_level > 70:
            additional_recommendations = [
                'Use white noise or nature sounds',
                'Practice progressive muscle relaxation',
                'Consider aromatherapy with lavender'
            ]
        else:
            additional_recommendations = [
                'Maintain consistent sleep schedule',
                'Ensure room is cool and dark',
                'Use comfortable bedding'
            ]
            
        return {
            'schedule': sleep_schedule,
            'wind_down_routine': wind_down_routine,
            'recommendations': additional_recommendations,
            'notes': self._generate_sleep_notes(profile, metrics)
        }

    def _create_stress_management_plan(self, metrics: HealthMetrics) -> Dict:
        """Generate stress management recommendations"""
        stress_level = 'high' if metrics.stress_level > 70 else 'moderate' if metrics.stress_level > 40 else 'low'
        
        return {
            'stress_level': stress_level,
            'recommended_activities': self.stress_relief_activities[stress_level],
            'daily_practices': [
                {'time': 'Morning', 'activity': 'Deep breathing exercises'},
                {'time': 'Afternoon', 'activity': 'Short meditation break'},
                {'time': 'Evening', 'activity': 'Relaxation routine'}
            ],
            'notes': self._generate_stress_notes(metrics)
        }

    def _calculate_sleep_duration(self, profile: HealthProfile, metrics: HealthMetrics) -> str:
        """Calculate optimal sleep duration based on profile and metrics"""
        base_duration = 8  # Base hours of sleep
        
        # Adjust for age
        if profile.age < 18:
            base_duration += 1
        elif profile.age > 65:
            base_duration -= 1
            
        # Adjust for activity level
        if profile.activity_level in ['very_active', 'moderate']:
            base_duration += 0.5
            
        # Adjust for stress
        if metrics.stress_level > 70:
            base_duration += 0.5
            
        return f"{base_duration}-{base_duration + 1} hours"

    def _calculate_hydration_needs(self, profile: HealthProfile, metrics: HealthMetrics) -> Dict:
        """Calculate daily hydration needs"""
        # Base calculation (in liters)
        base_fluid = profile.weight * 0.033  # 33ml per kg of body weight
        
        # Adjust for activity level
        activity_multiplier = {
            'sedentary': 1.0,
            'light': 1.1,
            'moderate': 1.2,
            'very_active': 1.4
        }
        
        adjusted_fluid = base_fluid * activity_multiplier[profile.activity_level]
        
        # Additional if stressed or active
        if metrics.stress_level > 70 or metrics.activity_score > 80:
            adjusted_fluid += 0.5
            
        return {
            'daily_target': f"{adjusted_fluid:.1f} liters",
            'recommendations': [
                'Drink 1 glass upon waking',
                'Carry a water bottle throughout the day',
                'Drink before, during, and after exercise',
                'Set hydration reminders every 2 hours'
            ]
        }

    def _generate_exercise_notes(self, profile: HealthProfile, metrics: HealthMetrics) -> List[str]:
        """Generate exercise-specific notes and cautions"""
        notes = []
        if metrics.readiness_score < 60:
            notes.append("Focus on light activities today due to lower readiness score")
        if metrics.stress_level > 70:
            notes.append("Incorporate more stress-relieving exercises like yoga or walking")
        if metrics.heart_rate > 100:
            notes.append("Monitor heart rate during exercise and adjust intensity accordingly")
        return notes

    def _generate_diet_notes(self, profile: HealthProfile, metrics: HealthMetrics) -> List[str]:
        """Generate diet-specific notes and recommendations"""
        notes = []
        if metrics.stress_level > 70:
            notes.append("Include more stress-reducing foods rich in omega-3s and antioxidants")
        if metrics.activity_score > 80:
            notes.append("Increase protein intake to support higher activity levels")
        if profile.dietary_restrictions:
            notes.append(f"Menu adjusted for dietary restrictions: {', '.join(profile.dietary_restrictions)}")
        return notes

    def _generate_sleep_notes(self, profile: HealthProfile, metrics: HealthMetrics) -> List[str]:
        """Generate sleep-specific notes and recommendations"""
        notes = []
        if metrics.sleep_score < 70:
            notes.append("Focus on improving sleep quality through consistent bedtime routine")
        if metrics.stress_level > 70:
            notes.append("Consider additional relaxation techniques before bedtime")
        if profile.activity_level == 'very_active':
            notes.append("Ensure adequate cool-down period between exercise and bedtime")
        return notes

    def _generate_stress_notes(self, metrics: HealthMetrics) -> List[str]:
        """Generate stress management notes"""
        notes = []
        if metrics.emotional_intensity > 0.7:
            notes.append("Practice grounding techniques during intense emotional states")
        if metrics.stress_level > 70:
            notes.append("Consider scheduling regular relaxation breaks throughout the day")
        if metrics.readiness_score < 60:
            notes.append("Focus on restorative activities to improve overall readiness")
        return notes