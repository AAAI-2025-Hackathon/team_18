
import os
import streamlit as st
import math
from mood_streamlit import HealthAgent

class HealthAgentWithFeedback(HealthAgent):
    """
    add update_recommendations_based_on_feedback to HealthAgent Subclass
    """
    def update_recommendations_based_on_feedback(self, mood_delta, sleep_delta, activity_delta=None):
        """
        Update the health recommendations based on changes in mood, sleep, and activity.

        Parameters:
          - mood_delta: Difference in average mood (feedback - initial).
          - sleep_delta: Difference in average sleep score.
          - activity_delta: Difference in average activity score.

        Returns:
          - new_recommendations: A dictionary with updated recommendations.
        """
        new_recommendations = {}

        # Update sleep recommendations based on sleep improvement or decline
        if sleep_delta is not None:
            if sleep_delta < 0:
                new_recommendations['sleep'] = self.sleep_recommendations['poor'] + [
                    "Consider adjusting your bedtime routine, reducing screen time before bed, and trying relaxation techniques."
                ]
            else:
                new_recommendations['sleep'] = self.sleep_recommendations['good'] + [
                    "Keep up your excellent sleep habits!"
                ]
        else:
            new_recommendations['sleep'] = ["Sleep data not available. Maintain a consistent sleep schedule."]

        # Update activity recommendations based on activity change
        if activity_delta is not None:
            if activity_delta < 0:
                new_recommendations['activity'] = self.activity_recommendations['low'] + [
                    "Try increasing your daily step count or engaging in light exercise."
                ]
            else:
                new_recommendations['activity'] = self.activity_recommendations['high'] + [
                    "Great job! Consider mixing up your workout routine to keep it interesting."
                ]
        else:
            new_recommendations['activity'] = ["Activity data not available. Continue your current activity level."]

        # Update overall mood recommendations based on mood change
        if mood_delta < 0:
            new_recommendations['overall'] = [
                "Your overall mood appears to have decreased. Consider integrating mindfulness practices, a balanced diet, and possibly consulting a health professional."
            ]
        else:
            new_recommendations['overall'] = [
                "Your mood has improved. Continue with your current habits and consider setting new health goals."
            ]

        return new_recommendations
def evaluate_feedback(initial_features, feedback_features, predictor, health_agent):
    """
    The function  compares key metrics ( mood scores, sleep scores, and activity scores)
    from the initial and follow-up data to compute deltas (improvements or declines). Based on these differences, two things happen:

    1- A progress report (showing delta changes) is displayed so patients can see how their mood/health metrics have changed.
    2- The recommendations (for sleep, exercise, and diet) are updated based on the feedback to further guide the patient.

    """
    # get datasets using the trained model
    initial_predictions = predictor.predict_mood(initial_features)
    feedback_predictions = predictor.predict_mood(feedback_features)

    # compute  average mood changes , find the detla
    initial_avg_mood = initial_predictions['predicted_mood'].mean()
    feedback_avg_mood = feedback_predictions['predicted_mood'].mean()
    mood_delta = feedback_avg_mood - initial_avg_mood

    # compute average sleep score and find delta,
    if 'sleep_score' in initial_features.columns and 'sleep_score' in feedback_features.columns:
        initial_avg_sleep = initial_features['sleep_score'].mean()
        feedback_avg_sleep = feedback_features['sleep_score'].mean()
        sleep_delta = feedback_avg_sleep - initial_avg_sleep
    else:
        sleep_delta = None

    # compute average activity score and find delta,
    if 'activity_score' in initial_features.columns and 'activity_score' in feedback_features.columns:
        initial_avg_activity = initial_features['activity_score'].mean()
        if math.isnan(initial_avg_activity):
            initial_avg_activity = -100
        feedback_avg_activity = feedback_features['activity_score'].mean()
        if math.isnan(feedback_avg_activity):
            initial_avg_activity = -100
        activity_delta = feedback_avg_activity - initial_avg_activity
    else:
        activity_delta = None

    # # Display  progress on UI =
    # st.markdown("### Progress based on last recommendation")
    # st.write(f"**Average Mood Change:** {mood_delta:.2f}")
    # if sleep_delta is not None:
    #     st.write(f"**Sleep Score Change:** {sleep_delta:.2f}")
    # if activity_delta is not None:
    #     st.write(f"**Activity Score Change:** {activity_delta:.2f}")

    # Update prompt  based on the computed deltas
    updated_recommendations = health_agent.update_recommendations_based_on_feedback(
        mood_delta, sleep_delta, activity_delta
    )

    # Return  new recommendations and the progress metrics for further use
    progress_metrics = {
        'initial_avg_mood': initial_avg_mood,
        'feedback_avg_mood': feedback_avg_mood,
        'mood_delta': mood_delta,
        'sleep_delta': sleep_delta,
        'activity_delta': activity_delta
    }
    return updated_recommendations, progress_metrics