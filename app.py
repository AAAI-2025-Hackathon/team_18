import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Moodify",
    page_icon="🧠",
    layout="wide"
)



from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from src.emotions import EmotionAnalyzer
from src.voice_analyzer import VoiceAnalyzer
from src.facial_analyzer import FacialAnalyzer
from src.emotional_arc import EmotionalArcAnalyzer
from src.multimodal_fusion import MultiModalFusion
import os
import argparse
import streamlit as st

from feedback_evaluator import HealthAgentWithFeedback, evaluate_feedback
from mood_predictor import mood_predictor_result, load_sensor_data
from src.visualization import (
    create_emotion_radar,
    create_emotion_timeline,
    create_intensity_gauge,
    create_emotional_arc_plot,
    create_color_palette_preview,
    create_journal_emotion_calendar,
    create_journal_emotion_summary,
    create_journal_timeline
)
from src.utils import init_session_state, update_emotion_history, get_recent_history
from src.color_palette import ColorPaletteManager
from src.journal_manager import JournalManager
import pandas as pd
from datetime import datetime
from src.ui_personalizer import UIPersonalizer # Added import
from src.avatar_assistant import AvatarAssistant, HealthData # Added import
from src.mock_health_data import HealthDataSimulator # Added import
from src.health_plan_generator import HealthProfile # Added import
from datetime import time # Added import
from src.mood_recommendations import MoodRecommendationManager # Added import
import os # Added import




# Initialize session state
init_session_state(st)

# Initialize analyzers
text_analyzer = EmotionAnalyzer()
voice_analyzer = VoiceAnalyzer()
facial_analyzer = FacialAnalyzer()
arc_analyzer = EmotionalArcAnalyzer()
color_manager = ColorPaletteManager()
journal_manager = JournalManager()
fusion_analyzer = MultiModalFusion()
ui_personalizer = UIPersonalizer() # Added initialization
avatar_assistant = AvatarAssistant() # Added initialization
health_simulator = HealthDataSimulator() # Added initialization
mood_recommender = MoodRecommendationManager() # Added initialization


# Title and description
st.title("🧠 Moodify")


# Main analysis section
st.markdown("## Your Personalized Multi-Modal Emotional Progression Analyzer")

st.sidebar.markdown('<p class="big-font"> Welcome to Moodify!  </p>', unsafe_allow_html=True)
st.sidebar.markdown("""
    Unlock emotions with text, voice, facial expressions, and wearable data!
""")

#st.sidebar.markdown('<p class="big-font"> Your personalized mood analysis tool!  </p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="medium-font">Team: Samy Movassaghi, Nima Taherkhani, Siamak Yousefi</p> </p>', unsafe_allow_html=True)
# Create columns for different modalities
col1, col2, col3 = st.columns(3)

# Initialize results
text_result = None
voice_result = None
facial_result = None
combined_emotions = None  # Initialize combined_emotions

with col1:
    st.markdown("### 📝 Text Analysis")
    text_input = st.text_area(
        "Enter your text:",
        height=100,
        placeholder="Type or paste your text here..."
    )
    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing text..."):
                text_result = text_analyzer.analyze_text(text_input)
                if text_result is not None:
                    st.success("Text analysis complete!")
        else:
            st.warning("Please enter some text to analyze.")

with col2:
    st.markdown("### 🎤 Voice Analysis")
    if st.button("Record Voice"):
        with st.spinner("Recording audio..."):
            try:
                audio_data = voice_analyzer.record_audio()
                st.success("Recording complete!")

                with st.spinner("Analyzing voice..."):
                    voice_result = voice_analyzer.analyze_emotion(audio_data)
                    if voice_result:
                        st.success("Voice analysis complete!")

                        # Save and display audio
                        audio_file = voice_analyzer.save_audio(audio_data)
                        st.audio(audio_file)
            except Exception as e:
                st.error(f"Error during voice analysis: {str(e)}")

with col3:
    st.markdown("### 👤 Facial Analysis")
    if st.button("Capture Face"):
        try:
            with st.spinner("Capturing image..."):
                frame = facial_analyzer.capture_frame()

                if frame is not None:
                    faces = facial_analyzer.detect_faces(frame)

                    if len(faces) > 0:
                        with st.spinner("Analyzing facial expression..."):
                            facial_result = facial_analyzer.analyze_emotion(frame)

                            if facial_result:
                                processed_frame = facial_analyzer.draw_results(frame, faces, facial_result)
                                image_path = facial_analyzer.save_frame(processed_frame)

                                if image_path:
                                    st.image(image_path, caption="Analyzed Face", use_column_width=True)
                                    st.success("Facial analysis complete!")
                    else:
                        st.warning("No faces detected. Please try again.")
                else:
                    st.error("Failed to capture image.")
        except Exception as e:
            st.error(f"Error during facial analysis: {str(e)}")

# Combined Analysis Section
st.markdown("---")
st.markdown("## 🔄 Combined Analysis Results")

# Perform fusion if any analysis was done
if text_result is not None or voice_result is not None or facial_result is not None:
    with st.spinner("Combining analyses..."):
        # Get combined emotions
        combined_emotions = fusion_analyzer.fuse_emotions(
            text_result, voice_result, facial_result
        )

        # Get confidence scores
        confidence_scores = fusion_analyzer.get_confidence_scores(
            text_result, voice_result, facial_result
        )

        if combined_emotions is not None and not combined_emotions.empty:
            # Create columns for visualization
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Overall Emotion Distribution")
                radar_chart = create_emotion_radar(combined_emotions)
                st.plotly_chart(radar_chart, use_container_width=True)

            with col2:
                st.markdown("### Dominant Emotion")
                if not combined_emotions.empty:
                    dominant_emotion = combined_emotions.iloc[0]
                    st.markdown(f"## {text_analyzer.get_emotion_emoji(dominant_emotion['label'])} {dominant_emotion['label'].title()}")
                    st.plotly_chart(
                        create_intensity_gauge(dominant_emotion['score'], dominant_emotion['label']),
                        use_container_width=True
                    )

            with col3:
                st.markdown("### Modality Confidence")
                for modality, score in confidence_scores.items():
                    st.progress(score, text=f"{modality.title()}: {score:.2f}")

            # Update emotion history
            update_emotion_history(st, combined_emotions)

            # Update UI based on emotional state
            dominant_emotion = combined_emotions.iloc[0]
            emotion_label = dominant_emotion['label']
            emotion_intensity = dominant_emotion['score']

            # Apply UI personalization
            ui_personalizer.apply_preferences(emotion_label, emotion_intensity)

            # Display UI adaptation message
            st.markdown("---")
            st.markdown("### 🎨 UI Personalization")
            st.info(f"""
                Interface adapted to your emotional state: 
                {text_analyzer.get_emotion_emoji(emotion_label)} {emotion_label.title()}

                The colors, layout, and interactions have been personalized to enhance your experience.
                Feel free to continue analyzing your emotions to see how the interface adapts!
            """)

            # Show emotional timeline
            st.markdown("### Emotional Journey")
            recent_history = get_recent_history(st)
            if recent_history is not None and not recent_history.empty:
                timeline_chart = create_emotion_timeline(recent_history)
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
        else:
            st.warning("Could not combine analyses. Please try again.")
else:
    st.info("Use the analysis tools above to see combined results!")


# Health Insights Section
with st.expander("🤖 Health Insights Assistant"):
    st.markdown("## Your Personal Health Assistant")
    st.markdown("""
        Meet your AI health assistant! They analyze your health data from smart devices
        and provide personalized insights and recommendations.
    """)

    # Get mock health data
    daily_health = health_simulator.generate_daily_data()
    health_data = HealthData(**daily_health)

    # Create columns for avatar and interaction
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display avatar
        st.markdown("### Your Assistant")
        initial_message = "Hello! I'm your health assistant. How can I help you today?"
        avatar_state = avatar_assistant.get_avatar_state(avatar_assistant._calculate_overall_status(health_data))
        avatar_assistant.display_avatar(avatar_state, message=initial_message)

    with col2:
        # Chat interface
        st.markdown("### Chat with Your Assistant")
        user_input = st.text_input("Ask about your health metrics:", placeholder="E.g., How's my sleep quality?")

        if user_input:
            response = avatar_assistant.get_conversation_response(user_input, health_data)
            # Update avatar with response
            avatar_assistant.display_avatar(avatar_state, message=response)

    # Display health insights
    st.markdown("### Today's Health Insights")
    insights = avatar_assistant.analyze_health_data(health_data)

    # Display metrics in columns
    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown("#### Key Metrics")
        for metric, value in insights['overall_status'].items():
            st.progress(value, f"{metric.replace('_', ' ').title()}: {int(value * 100)}%")

    with metrics_col2:
        st.markdown("#### Analysis")
        for metric, analysis in insights['metrics_analysis'].items():
            st.markdown(f"**{metric.title()}**: {analysis}")

    # Display recommendations
    st.markdown("### Personalized Recommendations")
    for recommendation in insights['recommendations']:
        st.markdown(f"- {recommendation}")

    # Show weekly trends
    st.markdown("### Weekly Trends")
    trends = health_simulator.generate_trends()
    trend_data = pd.DataFrame(trends)
    st.line_chart(trend_data)

    # Create mock user profile for demo
    user_profile = HealthProfile(
        height=175.0,  # cm
        weight=70.0,   # kg
        age=30,
        gender="not specified",
        activity_level="moderate",
        primary_goal="stress_reduction",
        dietary_restrictions=["none"],
        preferred_exercise_types=["walking", "yoga"],
        wake_time=time(6, 30),
        sleep_time=time(22, 30)
    )
    st.markdown("## 📋 Progress Feedback based on your wearable data")
    st.markdown("""
        Based on your current metrics, emotional state, and preferences, 
        here are your personalized recommendations:
    """)




def create_streamlit_interface_with_feedback( args):
    #st.title("Sleep Quality Predictor")

    # Initialize HealthAgent and load mood predictor results
    health_agent = HealthAgentWithFeedback()
    predictor, predictions, analysis = mood_predictor_result()
    data_dir = "/Users/samanehmovassaghi/Downloads/EmoSenseAI/src/ifh_affect_short"

    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #data_dir = os.path.join(current_dir, 'ifh_affect_short')

    samsung_data, oura_data = load_sensor_data(data_dir)


    # Create tabs for the UI
    # tab1, tab2, tab3 = st.tabs([
    #     "Predictions",  "Progress Feedback", "Chat Assistant"
    # ])
    plan_tabs = st.tabs(["Predicted status",  "Progress Feedback", "AI-based Health Guide"])

    with plan_tabs[0]:
    # ---------------------- Tab 2: Predictions ----------------------
        st.header("Mood Predictions")
        st.write(predictions.head())


    # ---------------------- Tab 5: progress feedback  ----------------------
    with plan_tabs[1]:
        st.header(" feedback data ")
        # First,extract results from past days
        features = predictor.extract_mood_features(
            # Replace these dummy parameters with the actual sensor data dictionaries.
            samsung_data=samsung_data,
            oura_data=oura_data
        )

        # Ensure that the features DataFrame has a "date" column.
        if 'date' not in features.columns:
            st.error("Error: The features DataFrame does not contain a 'date' column. Check extract_mood_features().")
        else:
            features = features.sort_values('date')
            unique_dates = features['date'].unique()

            # Default health parameters (from analysis or set defaults)
            health_params = {
                'sleep_score': analysis.get('oura_data', {}).get('sleep_score', 75),
                'readiness': analysis.get('oura_data', {}).get('readiness_score', 80),
                'activity_score': analysis.get('oura_data', {}).get('activity_score', 70),
                'hrv': analysis.get('oura_data', {}).get('hrv', 50),
                'steps': analysis.get('samsung_data', {}).get('steps', 8000),
                'heart_rate': analysis.get('samsung_data', {}).get('heart_rate', 70),
                'sleep_duration': analysis.get('samsung_data', {}).get('sleep_duration', 7)
            }

            # Determine the chatbot prompt based on available feedback data.
            if len(unique_dates) < 2:
                # if there is not enough days to provide feedbac
                st.info("Not enough data for feedback; using default recommendations.")
                chat_prompt = "Based on your current health metrics, please provide health recommendations."
            else:
                # Use the last two dates: second-to-last as initial, and last as feedback.
                initial_date = unique_dates[-2]
                feedback_date = unique_dates[-1]
                initial_features = features[features['date'] == initial_date]
                feedback_features = features[features['date'] == feedback_date]

                st.write(f"Initial Data Date: {initial_date}")
                st.write(f"Feedback Data Date: {feedback_date}")

                # Evaluate feedback and update recommendations.
                updated_recommendations, progress_metrics = evaluate_feedback(initial_features, feedback_features,
                                                                              predictor, health_agent)

                # Build an updated chatbot prompt with feedback deltas.
                chat_prompt = (
                    f"Your mood improved by {progress_metrics['mood_delta']:.2f} points, "
                    f"sleep score changed by {progress_metrics.get('sleep_delta', 0):.2f}, and "
                    f"activity score changed by {progress_metrics.get('activity_delta', 0):.2f}. "
                    "Based on these changes, please provide updated health recommendations."
                )
            # provide detailed progress metrics
            st.markdown("#### Detailed Progress Metrics")
            st.write(progress_metrics)
            st.markdown("#### Updated Recommendations Based on Your Progress")
            for key, recs in updated_recommendations.items():
                st.markdown(f"**{key.capitalize()} Recommendations:**")
                for rec in recs:
                    st.write(f"- {rec}")


    # ---------------------- Tab 6: Chat Assistant ----------------------
    with plan_tabs[2]:
        # get the updated recommendation from chatbot
        chat_response = health_agent.get_chat_response(chat_prompt, health_params)
        st.markdown("#### Chatbot Response:")
        st.write(chat_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sensor path')
    parser.add_argument('--data_path', type=str, default='ifh_affect', help='Mode of operation: train or generate')
    parser.add_argument('--par_ID', type=int, default=1)
    parser.add_argument('--start_date', type=str, default='2020-03-20', help='date on which analysis should start')
    parser.add_argument('--end_date', type=str, default='2020-03-26', help='date on which analysis should end')
    args = parser.parse_args()

    # Uncomment and set these if needed:
    # OURA_PATH = args.data_path + '/par_' + str(args.par_ID) + '/oura/'
    # SAMSUNG_PATH = args.data_path + '/par_' + str(args.par_ID) + '/samsung/'
    OURA_PATH ="/Users/samanehmovassaghi/Downloads/EmoSenseAI/src/ifh_affect_short/par_1/oura"
    
    SAMSUNG_PATH = "/Users/samanehmovassaghi/Downloads/EmoSenseAI/src/ifh_affect_short/par_1/samsung"
    st.sidebar.markdown("## Directory Contents")
    if os.path.exists(OURA_PATH):
        st.sidebar.write("Oura directory contents:")
        st.sidebar.write([f for f in os.listdir(OURA_PATH) if f.endswith('.csv')])
    if os.path.exists(SAMSUNG_PATH):
        st.sidebar.write("Samsung directory contents:")
        st.sidebar.write([f for f in os.listdir(SAMSUNG_PATH) if f.endswith('.csv')])

    # Create the Streamlit interface
    create_streamlit_interface_with_feedback( args)


    # Generate personalized plans
    st.markdown("## 📋 Your Personalized Health Plans")
    st.markdown("""
        Based on your current metrics, emotional state, and preferences, 
        here are your personalized recommendations:
    """)

    plans = avatar_assistant.generate_personalized_plans(health_data, user_profile)

    # Display plans in tabs
    plan_tabs = st.tabs(["Exercise", "Diet", "Sleep", "Stress Management"])

    with plan_tabs[0]:
        st.markdown("### 🏃‍♂️ Exercise Plan")
        exercise_plan = plans['exercise']

        # Display weekly schedule
        st.markdown("#### Weekly Schedule")
        cols = st.columns(3)
        for i, (day, schedule) in enumerate(exercise_plan['weekly_schedule'].items()):
            with cols[i % 3]:
                st.markdown(f"**{day}: {schedule['focus']}**")
                st.markdown(f"Intensity: {schedule['intensity'].title()}")
                st.markdown(f"Duration: {schedule['duration']}")
                st.markdown("Activities:")
                for activity in schedule['suggested_activities']:
                    st.markdown(f"- {activity}")
                st.markdown("---")

        # Display notes
        if exercise_plan['notes']:
            st.markdown("#### Important Notes")
            for note in exercise_plan['notes']:
                st.markdown(f"- {note}")

    with plan_tabs[1]:
        st.markdown("### 🥗 Diet Plan")
        diet_plan = plans['diet']

        # Display meal schedule
        st.markdown("#### Meal Schedule")
        for meal, details in diet_plan['meal_schedule'].items():
            st.markdown(f"**{meal.title()}** - {details['timing']}")
            if 'components' in details:
                st.markdown("Components:")
                for component in details['components']:
                    st.markdown(f"- {component}")
            st.markdown("Suggestions:")
            for suggestion in details['suggestions']:
                st.markdown(f"- {suggestion}")
            st.markdown("---")

        # Display hydration
        st.markdown("#### 💧 Hydration")
        st.markdown(f"Daily Target: {diet_plan['hydration']['daily_target']}")
        for rec in diet_plan['hydration']['recommendations']:
            st.markdown(f"- {rec}")

        # Display notes
        if diet_plan['notes']:
            st.markdown("#### Important Notes")
            for note in diet_plan['notes']:
                st.markdown(f"- {note}")

    with plan_tabs[2]:
        st.markdown("### 😴 Sleep Plan")
        sleep_plan = plans['sleep']

        # Display sleep schedule
        st.markdown("#### Sleep Schedule")
        schedule = sleep_plan['schedule']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bedtime", schedule['target_bedtime'])
        with col2:
            st.metric("Wake Time", schedule['target_wake_time'])
        with col3:
            st.metric("Optimal Duration", schedule['optimal_duration'])

        # Display wind-down routine
        st.markdown("#### Wind-down Routine")
        for step in sleep_plan['wind_down_routine']:
            st.markdown(f"**{step['time']}**: {step['activity']}")

        # Display recommendations
        st.markdown("#### Recommendations")
        for rec in sleep_plan['recommendations']:
            st.markdown(f"- {rec}")

    with plan_tabs[3]:
        st.markdown("### 🧘‍♂️ Stress Management")
        stress_plan = plans['stress_management']

        # Display stress level and activities
        st.markdown(f"**Current Stress Level**: {stress_plan['stress_level'].title()}")

        st.markdown("#### Recommended Activities")
        for activity in stress_plan['recommended_activities']:
            st.markdown(f"- {activity}")

        st.markdown("#### Daily Practices")
        for practice in stress_plan['daily_practices']:
            st.markdown(f"**{practice['time']}**: {practice['activity']}")

        # Display notes
        if stress_plan['notes']:
            st.markdown("#### Important Notes")
            for note in stress_plan['notes']:
                st.markdown(f"- {note}")

# Move mood recommendations outside the health insights expander
# Add new section after health insights expander
st.markdown("---")  # Add separator
st.markdown("## 🎵 Mood-Based Recommendations")
st.markdown("""
    Based on your emotional state and the time of day, 
    here are personalized music and meditation recommendations to support your wellbeing.
""")

# Get current emotional state and time of day
current_emotion = "neutral"  # Default to neutral
if combined_emotions is not None and not combined_emotions.empty:
    current_emotion = combined_emotions.iloc[0]['label'].lower()

time_of_day = mood_recommender.get_time_of_day()

# Get recommendations
recommendations = mood_recommender.get_recommendations(
    emotional_state=current_emotion,
    time_of_day=time_of_day
)

# Display recommendations in tabs
audio_tabs = st.tabs(["🧘‍♂️ Meditation", "🎵 Music"])

with audio_tabs[0]:
    st.markdown("### Recommended Meditations")
    if recommendations['meditation']:
        cols = st.columns(2)
        for i, meditation in enumerate(recommendations['meditation']):
            with cols[i % 2]:
                st.container()
                st.markdown(f"#### {meditation.title}")
                st.markdown(f"*Duration: {mood_recommender.format_duration(meditation.duration)}*")
                st.markdown(f"**Description**: {meditation.description}")
                st.markdown(f"**Mood Tags**: {', '.join(meditation.mood_tags)}")
                st.markdown(f"**Intensity**: {meditation.intensity.title()}")
                st.markdown(f"**Best Time**: {meditation.recommended_time.title()}")
                if os.path.exists(meditation.audio_file):
                    try:
                        with open(meditation.audio_file, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/wav')
                    except Exception as e:
                        st.error(f"Unable to play audio: {str(e)}")
                else:
                    st.warning("Audio file not available")
                st.markdown("---")
    else:
        st.info("No meditation recommendations available for your current state.")

with audio_tabs[1]:
    st.markdown("### Recommended Music")
    if recommendations['music']:
        cols = st.columns(2)
        for i, music in enumerate(recommendations['music']):
            with cols[i % 2]:
                st.container()
                st.markdown(f"#### {music.title}")
                st.markdown(f"*Duration: {mood_recommender.format_duration(music.duration)}*")
                st.markdown(f"**Description**: {music.description}")
                st.markdown(f"**Mood Tags**: {', '.join(music.mood_tags)}")
                st.markdown(f"**Intensity**: {music.intensity.title()}")
                st.markdown(f"**Best Time**: {music.recommended_time.title()}")
                if os.path.exists(music.audio_file):
                    try:
                        with open(music.audio_file, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/wav')
                    except Exception as e:
                        st.error(f"Unable to play audio: {str(e)}")
                else:
                    st.warning("Audio file not available")
                st.markdown("---")
    else:
        st.info("No music recommendations available for your current state.")

# Add duration preference slider
st.markdown("### Customize Recommendations")
duration_pref = st.slider(
    "Maximum Duration (minutes)",
    min_value=5,
    max_value=60,
    value=30,
    step=5
)

# Add refresh button
if st.button("Refresh Recommendations"):
    recommendations = mood_recommender.get_recommendations(
        emotional_state=current_emotion,
        time_of_day=time_of_day,
        duration_preference=duration_pref
    )


# Emotional Arc Section
with st.expander("Emotional Arc Analysis"):
    st.markdown("## Emotional Arc Analysis")
    st.markdown("""
        Track how emotions evolve over time with advanced arc mapping.
        This visualization shows emotional intensity, trends, and significant shifts.
    """)

    # Get analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Analysis Window Size", 3, 15, 5)
    with col2:
        shift_threshold = st.slider("Shift Detection Threshold", 0.1, 0.5, 0.3)

    # Get recent history and analyze
    recent_history = get_recent_history(st)
    if recent_history is not None and not recent_history.empty:
        arc_df = arc_analyzer.analyze_emotional_arc(recent_history, window_size)
        shifts = arc_analyzer.detect_emotional_shifts(arc_df, shift_threshold)

        # Display emotional arc plot
        arc_plot = create_emotional_arc_plot(arc_df, shifts)
        if arc_plot:
            st.plotly_chart(arc_plot, use_container_width=True)

            # Display detected shifts
            if shifts:
                st.markdown("### Detected Emotional Shifts")
                for shift in shifts:
                    direction = "positive" if shift['shift_magnitude'] > 0 else "negative"
                    st.markdown(f"""
                        - **{direction.title()} shift** at {shift['timestamp'].strftime('%H:%M:%S')}
                        - Magnitude: {abs(shift['shift_magnitude']):.2f}
                    """)
    else:
        st.info("Add more emotional data through analysis to see your emotional arc!")


# Color Palette Section
with st.expander("Color Palette Creator"):
    st.markdown("## Color Palette Creator")
    st.markdown("""
        Customize colors for different emotions and see how they affect the visualization.
        Select an emotion and use the color picker to choose your preferred color.
    """)

    # Emotion selection and color customization
    col1, col2 = st.columns(2)

    with col1:
        selected_emotion = st.selectbox(
            "Select emotion to customize:",
            list(color_manager.current_palette.keys())
        )

        current_color = color_manager.get_emotion_color(selected_emotion)
        new_color = st.color_picker(
            "Choose color",
            current_color
        )

        if st.button("Update Color"):
            color_manager.update_emotion_color(selected_emotion, new_color)
            st.success(f"Updated color for {selected_emotion}")

        if st.button("Reset to Default"):
            color_manager.reset_to_default()
            st.success("Reset all colors to default")

    with col2:
        # Show color palette preview
        st.markdown("### Color Palette Preview")
        intensity = st.slider("Emotion Intensity", 0.0, 1.0, 1.0, step=0.1)

        # Display color variants
        color_scheme = color_manager.get_color_scheme(selected_emotion)

        # Create columns for color swatches
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Primary**")
            st.markdown(
                f'<div style="background-color: {color_scheme["primary"]}; '
                f'padding: 20px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown("**Light Variant**")
            st.markdown(
                f'<div style="background-color: {color_scheme["light"]}; '
                f'padding: 20px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )

        with c3:
            st.markdown("**Dark Variant**")
            st.markdown(
                f'<div style="background-color: {color_scheme["dark"]}; '
                f'padding: 20px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )

    # Show preview of visualizations with custom colors
    st.markdown("### Visualization Preview")

    # Create sample data for preview
    sample_emotions = pd.DataFrame({
        'label': [selected_emotion, 'neutral'],
        'score': [0.8, 0.2],
        'timestamp': [datetime.now(), datetime.now()]
    })

    col1, col2 = st.columns(2)

    with col1:
        radar_chart = create_emotion_radar(sample_emotions)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        intensity_gauge = create_intensity_gauge(0.8, selected_emotion)
        st.plotly_chart(intensity_gauge, use_container_width=True)

    # Show full color palette preview
    st.markdown("### Complete Color Scheme")
    palette_preview = create_color_palette_preview(selected_emotion)
    st.plotly_chart(palette_preview, use_container_width=True)


# Journal Section
with st.expander("Interactive Emotion Journal"):
    st.markdown("## 📔 Interactive Emotion Journal")
    st.markdown("""
        Track your emotional journey and receive AI-powered insights about your patterns and trends.
        Add new entries and explore your emotional landscape over time.
    """)

    # Journal Entry Section
    st.markdown("### ✏️ New Journal Entry")
    journal_text = st.text_area(
        "What's on your mind?",
        height=150,
        placeholder="Write about your thoughts and feelings..."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        context = st.text_input("Context (optional)",
                               placeholder="e.g., Work, Family, Personal")
    with col2:
        tags = st.text_input("Tags (comma-separated)",
                            placeholder="e.g., meeting, project")

    if st.button("Analyze and Save Entry"):
        if journal_text:
            with st.spinner("Analyzing emotions in your entry..."):
                # Analyze emotions
                emotions_df = text_analyzer.analyze_text(journal_text)

                # Save to journal
                if journal_manager.add_entry(
                    journal_text,
                    emotions_df,
                    context,
                    [tag.strip() for tag in tags.split(',')] if tags else []
                ):
                    st.success("Journal entry saved successfully!")

                    # Show immediate analysis
                    if emotions_df is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Entry Analysis")
                            dominant_emotion = text_analyzer.get_dominant_emotion(emotions_df)
                            st.markdown(f"Primary emotion: {text_analyzer.get_emotion_emoji(dominant_emotion)} {dominant_emotion.title()}")

                        with col2:
                            st.plotly_chart(create_emotion_radar(emotions_df), use_container_width=True)
                else:
                    st.error("Failed to save journal entry. Please try again.")
        else:
            st.warning("Please write something in your journal entry.")

    # Journal Analysis Section
    st.markdown("---")
    st.markdown("### 📊 Journal Analytics")

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        days_filter = st.slider("Show entries from last N days", 1, 30, 7)
    with col2:
        emotion_filter = st.selectbox(
            "Filter by emotion",
            ["All"] + list(text_analyzer.emotion_map.keys())
        )

    # Get filtered entries
    filtered_entries = journal_manager.get_entries(
        start_date=datetime.now() - pd.Timedelta(days=days_filter),
        emotion_filter=emotion_filter if emotion_filter != "All" else None
    )

    if not filtered_entries.empty:
        # Display visualizations
        st.plotly_chart(create_journal_emotion_calendar(filtered_entries), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_journal_emotion_summary(filtered_entries), use_container_width=True)
        with col2:
            st.plotly_chart(create_journal_timeline(filtered_entries), use_container_width=True)

        # AI Insights
        st.markdown("### 🤖 AI Insights")
        insights = journal_manager.generate_insights(days_filter)

        st.markdown(f"**Summary**: {insights['summary']}")

        if insights['patterns']:
            st.markdown("**Emotional Patterns:**")
            for pattern in insights['patterns']:
                st.markdown(f"- {pattern}")

        if insights['recommendations']:
            st.markdown("**Recommendations:**")
            for recommendation in insights['recommendations']:
                st.markdown(f"- {recommendation}")

        # Journal Entries Table
        st.markdown("### 📝 Recent Entries")
        for _, entry in filtered_entries.iterrows():
            with st.container():
                st.write(f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry['primary_emotion'].title()}")
                st.markdown(f"**Context**: {entry['context'] if entry['context'] else 'Not specified'}")
                st.markdown(f"**Tags**: {', '.join(entry['tags']) if entry['tags'] else 'None'}")
                st.markdown(f"**Entry**: {entry['text']}")
                st.markdown("**Emotions detected:**")
                for emotion, score in entry['emotions'].items():
                    st.markdown(f"- {text_analyzer.get_emotion_emoji(emotion)} {emotion.title()}: {score:.2f}")
    else:
        st.info("No journal entries found. Start writing to see your emotional patterns!")

# Footer
st.markdown("---")

# # Simplify the VideoTransformer class to test basic functionality
# class VideoTransformer(VideoTransformerBase):
#     def transform(self, frame):
#         # Simply return the frame without any modifications
#         return frame.to_ndarray(format="bgr24")

# # Use webrtc_streamer with the simplified VideoTransformer
# webrtc_ctx = webrtc_streamer(
#     key="example",
#     video_processor_factory=VideoTransformer,
#     async_processing=True,
# )
# if webrtc_ctx.video_processor:
#     st.write("Face detection is active. Look at the camera!")



