import os
import argparse
import streamlit as st
from feedback_evaluator import HealthAgentWithFeedback, evaluate_feedback
from mood_predictor import mood_predictor_result, load_sensor_data

def create_streamlit_interface_with_feedback( args):
    st.title("Sleep Quality Predictor")

    # Initialize HealthAgent and load mood predictor results
    health_agent = HealthAgentWithFeedback()
    predictor, predictions, analysis = mood_predictor_result()


    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'ifh_affect_short')
    samsung_data, oura_data = load_sensor_data(data_dir)


    # Create tabs for the UI
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Input", "Predictions", "Visualizations",
        "Health Recommendations",  "Progress Feedback", "Chat Assistant"
    ])

    # ---------------------- Tab 1: Data Input ----------------------
    with tab1:
        st.header("Your Health Data")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Samsung Health Data")
            # (Place code here to show or input Samsung data)
        with col2:
            st.subheader("Oura Health Data")
            # (Place code here to show or input Oura data)

    # ---------------------- Tab 2: Predictions ----------------------
    with tab2:
        st.header("Mood Predictions")
        st.write(predictions.head())

    # ---------------------- Tab 3: Visualizations ----------------------
    with tab3:
        st.header("Visualizations")
        ## maybe place some fancy avatar or empji to show the current mode or suchh

    # ---------------------- Tab 4: Health Recommendations ----------------------
    with tab4:
        st.header("Initial Health Recommendations")
        st.write(analysis)

    # ---------------------- Tab 5: progress feedback  ----------------------
    with tab5:
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
    with tab6:
        # get the updated recommendation from chatbot
        chat_response = health_agent.get_chat_response(chat_prompt, health_params)
        st.markdown("#### Chatbot Response:")
        st.write(chat_response)
        print( "done done ")

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
    OURA_PATH =""
    SAMSUNG_PATH = ""
    st.sidebar.markdown("## Directory Contents")
    if os.path.exists(OURA_PATH):
        st.sidebar.write("Oura directory contents:")
        st.sidebar.write([f for f in os.listdir(OURA_PATH) if f.endswith('.csv')])
    if os.path.exists(SAMSUNG_PATH):
        st.sidebar.write("Samsung directory contents:")
        st.sidebar.write([f for f in os.listdir(SAMSUNG_PATH) if f.endswith('.csv')])

    # Create the Streamlit interface
    create_streamlit_interface_with_feedback( args)