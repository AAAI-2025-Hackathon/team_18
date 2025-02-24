import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from src.health_plan_generator import HealthPlanGenerator, HealthProfile, HealthMetrics

@dataclass
class HealthData:
    heart_rate: float
    sleep_score: float
    readiness_score: float
    activity_score: float
    stress_level: float
    timestamp: datetime

class AvatarAssistant:
    def __init__(self):
        self.avatar_states = {
            'happy': '''
            <div style="text-align: center; background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px;">
                <div class="speech-bubble" style="
                    position: relative;
                    background: #ffffff;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    animation: fadeIn 0.5s ease-in;
                ">
                    <div class="message" style="font-size: 16px;"></div>
                </div>
                <svg width="100" height="100" viewBox="0 0 100 100" style="display: block; margin: auto;">
                    <circle cx="50" cy="50" r="45" fill="#FFD700" stroke="#000" stroke-width="2">
                        <animate attributeName="r" values="45;47;45" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="35" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;38;40" dur="1s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;5.2;5" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="65" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;38;40" dur="1s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;5.2;5" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <path d="M 30 60 Q 50 80 70 60" fill="none" stroke="#000" stroke-width="3">
                        <animate attributeName="d" 
                            values="M 30 60 Q 50 80 70 60;M 30 60 Q 50 85 70 60;M 30 60 Q 50 80 70 60" 
                            dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="stroke-width" values="3;3.5;3" dur="2s" repeatCount="indefinite"/>
                    </path>
                </svg>
                <style>
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    @keyframes morphTransition {
                        from { transform: scale(1) rotate(0deg); }
                        50% { transform: scale(1.05) rotate(2deg); }
                        to { transform: scale(1) rotate(0deg); }
                    }
                    svg {
                        transition: all 0.5s ease-in-out;
                        animation: morphTransition 0.5s ease-in-out;
                    }
                </style>
            </div>
            ''',
            'neutral': '''
            <div style="text-align: center; background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px;">
                <div class="speech-bubble" style="
                    position: relative;
                    background: #ffffff;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    animation: fadeIn 0.5s ease-in;
                ">
                    <div class="message" style="font-size: 16px;"></div>
                </div>
                <svg width="100" height="100" viewBox="0 0 100 100" style="display: block; margin: auto;">
                    <circle cx="50" cy="50" r="45" fill="#FFD700" stroke="#000" stroke-width="2">
                        <animate attributeName="r" values="45;46;45" dur="3s" repeatCount="indefinite"/>
                        <animate attributeName="fill-opacity" values="1;0.95;1" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="35" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;41;40" dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;5.1;5" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="65" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;41;40" dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;5.1;5" dur="2s" repeatCount="indefinite"/>
                    </circle>
                    <line x1="30" y1="65" x2="70" y2="65" stroke="#000" stroke-width="3">
                        <animate attributeName="y1" values="65;66;65" dur="3s" repeatCount="indefinite"/>
                        <animate attributeName="y2" values="65;66;65" dur="3s" repeatCount="indefinite"/>
                        <animate attributeName="stroke-width" values="3;3.2;3" dur="2s" repeatCount="indefinite"/>
                    </line>
                </svg>
                <style>
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    @keyframes morphTransition {
                        from { transform: scale(1) rotate(0deg); }
                        50% { transform: scale(1.02) rotate(1deg); }
                        to { transform: scale(1) rotate(0deg); }
                    }
                    svg {
                        transition: all 0.5s ease-in-out;
                        animation: morphTransition 0.5s ease-in-out;
                    }
                </style>
            </div>
            ''',
            'concerned': '''
            <div style="text-align: center; background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px;">
                <div class="speech-bubble" style="
                    position: relative;
                    background: #ffffff;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    animation: fadeIn 0.5s ease-in;
                ">
                    <div class="message" style="font-size: 16px;"></div>
                </div>
                <svg width="100" height="100" viewBox="0 0 100 100" style="display: block; margin: auto;">
                    <circle cx="50" cy="50" r="45" fill="#FFD700" stroke="#000" stroke-width="2">
                        <animate attributeName="r" values="45;44;45" dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="fill-opacity" values="1;0.9;1" dur="1.5s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="35" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;42;40" dur="1.5s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;4.8;5" dur="1.5s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="65" cy="40" r="5" fill="#000">
                        <animate attributeName="cy" values="40;42;40" dur="1.5s" repeatCount="indefinite"/>
                        <animate attributeName="r" values="5;4.8;5" dur="1.5s" repeatCount="indefinite"/>
                    </circle>
                    <path d="M 30 70 Q 50 50 70 70" fill="none" stroke="#000" stroke-width="3">
                        <animate attributeName="d" 
                            values="M 30 70 Q 50 50 70 70;M 30 70 Q 50 48 70 70;M 30 70 Q 50 50 70 70" 
                            dur="2s" repeatCount="indefinite"/>
                        <animate attributeName="stroke-width" values="3;2.8;3" dur="1.5s" repeatCount="indefinite"/>
                    </path>
                </svg>
                <style>
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    @keyframes morphTransition {
                        from { transform: scale(1) rotate(0deg); }
                        50% { transform: scale(0.98) rotate(-1deg); }
                        to { transform: scale(1) rotate(0deg); }
                    }
                    svg {
                        transition: all 0.5s ease-in-out;
                        animation: morphTransition 0.5s ease-in-out;
                    }
                </style>
            </div>
            '''
        }
        self.health_plan_generator = HealthPlanGenerator()

    def get_avatar_state(self, health_metrics: Dict[str, float]) -> str:
        """Determine avatar state based on health metrics"""
        avg_score = sum(health_metrics.values()) / len(health_metrics)
        if avg_score >= 0.7:
            return 'happy'
        elif avg_score >= 0.4:
            return 'neutral'
        else:
            return 'concerned'

    def display_avatar(self, state: str, message: Optional[str] = None):
        """Display the avatar in specified state with optional message"""
        avatar_svg = self.avatar_states.get(state, self.avatar_states['neutral'])

        # If there's a message, insert it into the speech bubble
        if message:
            avatar_svg = avatar_svg.replace('<div class="message" style="font-size: 16px;"></div>', 
                                          f'<div class="message" style="font-size: 16px;">{message}</div>')

        # Display the avatar with the speech bubble
        st.components.v1.html(avatar_svg, height=250)

    def analyze_health_data(self, health_data: HealthData) -> Dict[str, Any]:
        """Analyze health data and generate insights"""
        insights = {
            'overall_status': self._calculate_overall_status(health_data),
            'recommendations': self._generate_recommendations(health_data),
            'metrics_analysis': self._analyze_metrics(health_data)
        }
        return insights

    def _calculate_overall_status(self, data: HealthData) -> Dict[str, float]:
        """Calculate overall health status"""
        return {
            'sleep_quality': data.sleep_score / 100,
            'physical_readiness': data.readiness_score / 100,
            'activity_level': data.activity_score / 100,
            'stress_management': 1 - (data.stress_level / 100)
        }

    def _generate_recommendations(self, data: HealthData) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # Sleep recommendations
        if data.sleep_score < 70:
            recommendations.append(
                "Your sleep quality could be improved. Consider going to bed earlier "
                "and maintaining a consistent sleep schedule."
            )

        # Stress management
        if data.stress_level > 60:
            recommendations.append(
                "Your stress levels are elevated. Try incorporating meditation "
                "or deep breathing exercises into your daily routine."
            )

        # Activity recommendations
        if data.activity_score < 60:
            recommendations.append(
                "Your activity level is lower than optimal. Try to include more "
                "movement in your day, even short walks can make a difference."
            )

        # Readiness-based recommendations
        if data.readiness_score < 70:
            recommendations.append(
                "Your body's recovery signals indicate you might need more rest. "
                "Consider a lighter workout today and focus on recovery."
            )

        return recommendations

    def _analyze_metrics(self, data: HealthData) -> Dict[str, str]:
        """Provide detailed analysis of individual metrics"""
        analysis = {}

        # Heart rate analysis
        if data.heart_rate < 60:
            analysis['heart_rate'] = "Your resting heart rate is low, indicating good cardiovascular fitness."
        elif data.heart_rate < 100:
            analysis['heart_rate'] = "Your heart rate is within a normal range."
        else:
            analysis['heart_rate'] = "Your heart rate is elevated. This might be due to stress or recent activity."

        # Sleep analysis
        if data.sleep_score >= 85:
            analysis['sleep'] = "Excellent sleep quality! Keep maintaining your current sleep routine."
        elif data.sleep_score >= 70:
            analysis['sleep'] = "Good sleep quality. Small improvements to your sleep routine could help optimize it further."
        else:
            analysis['sleep'] = "Your sleep quality could be improved. Let's work on your sleep hygiene."

        # Activity analysis
        if data.activity_score >= 80:
            analysis['activity'] = "Great job staying active! You're meeting your movement goals."
        elif data.activity_score >= 60:
            analysis['activity'] = "Moderate activity level. Try to incorporate more movement into your day."
        else:
            analysis['activity'] = "Your activity level is lower than recommended. Let's find ways to move more."

        return analysis

    def get_conversation_response(self, user_input: str, health_data: HealthData) -> str:
        """Generate conversational responses based on user input and health data"""
        # Simple keyword-based response system
        user_input = user_input.lower()

        response = ""
        if 'sleep' in user_input:
            if health_data.sleep_score >= 80:
                response = "Your sleep quality is excellent! You're getting good rest which helps with recovery and mental clarity."
            else:
                response = "I notice your sleep could be better. Would you like some tips for improving sleep quality?"

        elif 'stress' in user_input:
            if health_data.stress_level > 70:
                response = "I can see your stress levels are high. Let's work on some stress management techniques."
            else:
                response = "You're managing stress well! Keep up with your current stress management practices."

        elif 'activity' in user_input or 'exercise' in user_input:
            if health_data.activity_score >= 70:
                response = "You're doing great with staying active! Keep up the momentum."
            else:
                response = "I can suggest some ways to increase your daily activity level. Would you like to hear them?"

        elif 'tired' in user_input or 'fatigue' in user_input:
            if health_data.readiness_score < 60:
                response = "Your readiness score suggests you might need more rest. It's okay to take it easy today."
            else:
                response = "While you're feeling tired, your body's metrics actually look good. Maybe try some energizing activities?"

        else:
            response = "I'm here to help! You can ask me about your sleep, stress, activity, or general wellbeing."

        return response

    def generate_personalized_plans(self, health_data: HealthData, profile: HealthProfile) -> Dict:
        """Generate personalized health plans based on current metrics and user profile"""
        metrics = HealthMetrics(
            heart_rate=health_data.heart_rate,
            sleep_score=health_data.sleep_score,
            readiness_score=health_data.readiness_score,
            activity_score=health_data.activity_score,
            stress_level=health_data.stress_level,
            emotional_state=self.get_avatar_state(self._calculate_overall_status(health_data)),
            emotional_intensity=max(self._calculate_overall_status(health_data).values()),
            timestamp=health_data.timestamp
        )

        return self.health_plan_generator.generate_plans(profile, metrics)