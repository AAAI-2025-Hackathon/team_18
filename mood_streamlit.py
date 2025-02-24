from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
# import uvicorn
from datetime import datetime,timedelta
import os
from dotenv import load_dotenv
import json
import argparse

# import cudf
from openai import OpenAI
import openai


from mood_predictor import *

openai_api_key_input= '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'

# Load environment variables
load_dotenv()

app = FastAPI()


st.markdown("""
<style>
.big-font {
    font-size:30px !important;
.medium-font {
    font-size:20px !important;
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

#############################################################################
#Configuring the title and sidebar for the app created by streamlit


    


def get_health_data(predictor, args):
    """Get and merge health data"""
    # Load data
    # oura_df = load_oura_data()
    # samsung_df = load_samsung_data()

    ## for samsung
    samsung_sensor_list = ['samsung_awake',
        'samsung_hrv',
        'samsung_imu',
        'samsung_ppg_data',
        'samsung_pedometer_data']
    
    awake_file = os.path.join(SAMSUNG_PATH, 'awake_times.csv')
    hrv_file = os.path.join(SAMSUNG_PATH, 'hrv_1min.csv')
    imu_file = os.path.join(SAMSUNG_PATH, 'imu.csv')
    ppg_file = os.path.join(SAMSUNG_PATH, 'ppg.csv')
    pressure_file = os.path.join(SAMSUNG_PATH, 'pressure.csv')
    pedometer_file = os.path.join(SAMSUNG_PATH, 'pedometer.csv')

    samsung_awake_data = pd.read_csv(awake_file)
    samsung_hrv_data = pd.read_csv(hrv_file)
    # samsung_imu_data = pd.read_csv(imu_file)
    # samsung_ppg_data = pd.read_csv(ppg_file)
    samsung_pressure_data = pd.read_csv(pressure_file)
    samsung_pedometer_data = pd.read_csv(pedometer_file)

    
    samsung_df = predictor.preprocess_samsung_data(
        samsung_awake_data, samsung_hrv_data, samsung_pedometer_data
    )
    
    print('samsung', samsung_df.keys())

    ## do it for oura
    oura_sensor_list = ['oura_sleep_data',
        'oura_activity_data',
        'oura_readiness_data',
        'oura_heart_rate_data']

    oura_sleep_file = os.path.join(OURA_PATH, 'sleep.csv')
    oura_activity_file = os.path.join(OURA_PATH, 'activity.csv')
    oura_readiness_file = os.path.join(OURA_PATH, 'readiness.csv')
    oura_heart_rate_file = os.path.join(OURA_PATH, 'heart_rate.csv')

    oura_activity_data = pd.read_csv(oura_activity_file)
    oura_readiness_data = pd.read_csv(oura_readiness_file)
    oura_heart_rate_data = pd.read_csv(oura_heart_rate_file)
    oura_sleep_data = pd.read_csv(oura_sleep_file)
    
    oura_df = predictor.preprocess_oura_data(
        oura_sleep_data, oura_activity_data, oura_readiness_data, oura_heart_rate_data
    )
    print('oura', oura_df.keys())


    combined_df = predictor.engineer_features(samsung_df, oura_df)
    print('combined', combined_df.keys())

    # Debug: Show data shapes
    st.sidebar.write("Oura data shape:", oura_df.shape if not oura_df.empty else "Empty")
    st.sidebar.write("Samsung data shape:", samsung_df.shape if not samsung_df.empty else "Empty")
    
    # Convert dates to datetime if needed
    if not oura_df.empty:
        date_column = [col for col in oura_df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
        oura_df['date'] = pd.to_datetime(oura_df[date_column])
    
    if not samsung_df.empty:
        date_column = [col for col in samsung_df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
        samsung_df['date'] = pd.to_datetime(samsung_df[date_column])

    # print('date', samsung_df['date'][0:10])
    # print('date', oura_df['date'][0:10])

    ### select chunk of dataframes from start_date to end_date
    
    samsung_df['df'] = samsung_df['date'].sort_values()
    oura_df['df'] = oura_df['date'].sort_values()

    samsung_df = samsung_df[(samsung_df['date'] >= args.start_date) & (samsung_df['date'] <= args.end_date)]
    oura_df = oura_df[(oura_df['date'] >= args.start_date) & (oura_df['date'] <= args.end_date)]

    print('date', samsung_df['date'] )
    print('date', oura_df['date'])
    

    ### TODO 
    # Merge data if both dataframes have data
    if not oura_df.empty and not samsung_df.empty:
        merged_df = pd.merge(oura_df, samsung_df, on='date', how='outer')
        merged_df = merged_df.sort_values('date')
        st.sidebar.write("Merged data shape:", merged_df.shape)
    else:
        merged_df = pd.DataFrame()
        if oura_df.empty:
            st.warning("No Oura data found")
        if samsung_df.empty:
            st.warning("No Samsung data found")


    return {
        'oura_data': {
            'sleep_score': np.mean(oura_df['score_sleep']), # if not oura_df.empty and 'sleep_score' in oura_df.columns else 75,
            'readiness_score': np.mean(oura_df['score_temperature']), # if not oura_df.empty and 'readiness_score' in oura_df.columns else 80,
            'activity_score': np.mean(oura_df['score_activity']), # if not oura_df.empty and 'activity_score' in oura_df.columns else 70,
            'deep_sleep': np.mean(oura_df['score_sleep_balance']), # if not oura_df.empty and 'deep_sleep' in oura_df.columns else 2,
            'rem_sleep': np.mean(oura_df['rem']), # if not oura_df.empty and 'rem_sleep' in oura_df.columns else 1.5,
            'hrv': np.mean(oura_df['hr_average']), # if not oura_df.empty and 'hrv' in oura_df.columns else 50
        },
        'samsung_data': {
            'steps': np.mean(samsung_df['num_total_steps']), # if not samsung_df.empty and 'steps' in samsung_df.columns else 8000,
            'heart_rate': np.mean(samsung_df['hr_mean']), # if not samsung_df.empty and 'heart_rate' in samsung_df.columns else 70,

            'sleep_duration': np.mean(samsung_df['total_awake_hours'] ), # if not samsung_df.empty and 'sleep_duration' in samsung_df.columns else 7,
            # 'stress_level': np.mean( samsung_df['stress_level'] ), # if not samsung_df.empty and 'stress_level' in samsung_df.columns else 30
            'cal_burn_kcal' : np.mean(samsung_df['cal_burn_kcal']),
  
        },
        'merged_df': merged_df
    }
    




def check_data_structure(file_path):
    """Debug function to check JSON structure"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            st.sidebar.write(f"Data structure for {os.path.basename(file_path)}:")
            st.sidebar.json(data)
    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")



        
class HealthAgent:
    def __init__(self):
        self.external_sources = {
            "Mayo Clinic": "https://www.mayoclinic.org/healthy-lifestyle/adult-health/in-depth/sleep/art-20048379",
            "Sleep Foundation": "https://www.sleepfoundation.org/sleep-hygiene",
            "CDC": "https://www.cdc.gov/sleep/about_sleep/sleep_hygiene.html"
        }
        
        self.recommendations = {
            'Mayo Clinic': {
                'sleep': [
                    "Stick to a sleep schedule",
                    "Pay attention to what you eat and drink",
                    "Create a restful environment",
                    "Limit daytime naps",
                    "Include physical activity in your daily routine"
                ],
                # 'stress': [
                #     "Regular exercise",
                #     "Relaxation techniques",
                #     "Maintain social connections",
                #     "Set realistic goals",
                #     "Practice stress management"
                # ]
            },
            'Sleep Foundation': {
                'sleep': [
                    "Optimize your bedroom environment",
                    "Follow a consistent pre-bed routine",
                    "Use comfortable bedding",
                    "Avoid bright lights before bedtime",
                    "Practice relaxation techniques"
                ],
                'exercise': [
                    "Exercise regularly, but not too close to bedtime",
                    "Get exposure to natural daylight",
                    "Stay active throughout the day",
                    "Balance intensity of workouts",
                    "Include both cardio and strength training"
                ]
            },
            'CDC': {
                'sleep': [
                    "Be consistent with sleep schedule",
                    "Make sure bedroom is quiet, dark, and relaxing",
                    "Remove electronic devices from bedroom",
                    "Avoid large meals before bedtime",
                    "Get enough natural light during the day"
                ],
                'health': [
                    "Maintain a healthy diet",
                    "Stay physically active",
                    "Manage chronic conditions",
                    "Practice good sleep hygiene",
                    "Regular health check-ups"
                ]
            }
        }

        self.sleep_recommendations = {
            'poor': [
                "Establish a consistent sleep schedule",
                "Create a relaxing bedtime routine",
                "Limit screen time before bed",
                "Ensure your bedroom is dark and cool",
                "Consider meditation or deep breathing exercises"
            ],
            'moderate': [
                "Maintain your current sleep schedule",
                "Try to increase deep sleep phases",
                "Monitor caffeine intake after noon",
                "Get regular exercise, but not too close to bedtime"
            ],
            'good': [
                "Keep up your excellent sleep habits",
                "Fine-tune your sleep environment",
                "Continue monitoring your sleep patterns"
            ]
        }
        
        self.activity_recommendations = {
            'low': [
                "Start with short walks throughout the day",
                "Try gentle stretching exercises",
                "Set achievable daily step goals",
                "Consider low-impact activities like swimming",
                "Gradually increase activity duration"
            ],
            'moderate': [
                "Mix cardio and strength training",
                "Aim for 150 minutes of moderate activity per week",
                "Include flexibility exercises",
                "Try interval training",
                "Join group fitness classes"
            ],
            'high': [
                "Maintain your excellent activity level",
                "Ensure proper recovery between workouts",
                "Mix up your routine to prevent plateaus",
                "Consider training for an event",
                "Focus on form and technique"
            ]
        }
        
        self.suggested_questions = [
            "How can I improve my deep sleep?",
            "What's the ideal sleep schedule for my age?",
            # "How does stress affect my sleep quality?",
            "What's the relationship between exercise and sleep?",
            "How can I establish a better bedtime routine?",
            "What foods should I avoid before bedtime?",
            "How does screen time affect my sleep?",
            "What's the optimal bedroom temperature for sleep?",
            "How can I reduce sleep anxiety?",
            "Should I try meditation for better sleep?"
        ]

    def get_chat_response(self, prompt, health_params):
        """Generate response with recommendations from multiple sources"""
        response = "Here's what I found from multiple trusted sources:\n\n"

        ### SY
        sleep_score, readiness, activity_score, hrv, steps, heart_rate, sleep_duration = health_params

        prefix_prompt = """You are a professional health specialist. Based on the following health metrics, 
            provide specific, actionable advice about the user's daily life and well-being.

            User's Health Metrics:
            - Sleep Score: {sleep}/100
            - Readiness Score: {readiness}/100
            - Activity Score: {activity}/100
            - Heart Rate Variability (HRV): {hrv} ms
            - Daily Steps: {steps}
            - Average Heart Rate: {heart_rate} bpm
            - Sleep Duration: {sleep_duration} hours

            Please analyze these metrics and provide:
            1. An overall assessment of the user's health status
            2. Specific recommendations for improvement
            3. Any areas of concern that should be addressed
            4. Positive habits that should be maintained

            Advice:""".format(
                sleep=health_params['sleep_score'],
                readiness=health_params['readiness'],
                activity=health_params['activity_score'],
                hrv=health_params['hrv'],
                steps=health_params['steps'],
                heart_rate=health_params['heart_rate'],
                sleep_duration=health_params['sleep_duration']
            )


        ### openai chat completion
        model ='gpt-4'
        os.environ["OPENAI_API_KEY"] = openai_api_key_input
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
       
        # Creating a message as required by the API
        messages = [{"role": "user", "content": prefix_prompt + prompt}]
           # Calling the ChatCompletion API
        response_openai = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        # Returning the extracted response
        response+='\n'+ response_openai.choices[0].message.content
        response += '\n'
        print('response', response)

        
        
        # Add recommendations based on keywords
        keywords = {
            'sleep': ['sleep', 'bed', 'rest', 'nap', 'insomnia'],
            # 'stress': ['stress', 'anxiety', 'worried', 'tension'],
            'exercise': ['exercise', 'activity', 'workout', 'fitness'],
            'health': ['health', 'wellness', 'lifestyle', 'habits']
        }
        
        # Find matching categories based on prompt
        matching_categories = []
        for category, terms in keywords.items():
            if any(term in prompt.lower() for term in terms):
                matching_categories.append(category)
        
        # If no matches, default to sleep category
        if not matching_categories:
            matching_categories = ['sleep']
        
        # Add recommendations from each source
        for source, recommendations in self.recommendations.items():
            response += f"\n**{source} Recommendations:**\n"
            for category in matching_categories:
                if category in recommendations:
                    response += "\n".join([f"- {rec}" for rec in recommendations[category][:3]])
                    response += "\n"
        
        # Add references
        response += "\n**References:**\n"
        for source, url in self.external_sources.items():
            response += f"- [{source}]({url})\n"
        
        # Add suggested follow-up questions
        response += "\n**You might also want to ask:**\n"
        relevant_questions = [q for q in self.suggested_questions 
                            if any(cat in q.lower() for cat in matching_categories)]
        response += "\n".join([f"- {q}" for q in relevant_questions[:3]])
        
        return response

    def analyze_data(self, samsung_data, oura_data):
        """Analyze health data and generate insights"""
        analysis = {
            'sleep_quality': self._analyze_sleep(samsung_data, oura_data),
            'activity_level': self._analyze_activity(samsung_data, oura_data),
            'overall_health': self._analyze_overall_health(samsung_data, oura_data),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis

    def _analyze_sleep(self, samsung_data, oura_data):
        """Analyze sleep metrics"""
        sleep_score = oura_data.get('sleep_score', 0)
        sleep_duration = samsung_data.get('sleep_duration', 0)
        
        quality_assessment = {
            'score': sleep_score,
            'duration': sleep_duration,
            'quality': 'poor' if sleep_score < 70 else 'moderate' if sleep_score < 85 else 'good',
            'issues': []
        }
        
        # Identify potential issues
        if sleep_duration < 7:
            quality_assessment['issues'].append("Insufficient sleep duration")
        if sleep_score < 70:
            quality_assessment['issues'].append("Low sleep quality")
        
        return quality_assessment

    def _analyze_activity(self, samsung_data, oura_data):
        """Analyze activity metrics"""
        steps = samsung_data.get('steps', 0)
        activity_score = oura_data.get('activity_score', 0)
        
        activity_level = {
            'steps': steps,
            'score': activity_score,
            'level': 'low' if steps < 5000 else 'moderate' if steps < 10000 else 'high',
            'recommendations': []
        }
        
        # Add specific recommendations
        if steps < 5000:
            activity_level['recommendations'].extend(self.activity_recommendations['low'])
        elif steps < 10000:
            activity_level['recommendations'].extend(self.activity_recommendations['moderate'])
        else:
            activity_level['recommendations'].extend(self.activity_recommendations['high'])
        
        return activity_level

    def _analyze_overall_health(self, samsung_data, oura_data):
        """Analyze overall health status"""
        readiness_score = oura_data.get('readiness_score', 0)
        heart_rate = samsung_data.get('heart_rate', 0)
        # stress_level = samsung_data.get('stress_level', 0)
        
        
        return {
            'readiness': readiness_score,
            'heart_rate': heart_rate,
            # 'stress_level': stress_level,
            'status': 'good' if readiness_score > 80 else 'moderate' if readiness_score > 60 else 'poor'
        }

    def _generate_recommendations(self, analysis):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Sleep recommendations
        sleep_quality = analysis['sleep_quality']['quality']
        recommendations.extend(self.sleep_recommendations[sleep_quality])
        
        # Activity recommendations
        activity_level = analysis['activity_level']['level']
        recommendations.extend(self.activity_recommendations[activity_level])
        
        return recommendations



# Update the create_streamlit_interface function to use HealthAgent
def create_streamlit_interface(sleep_quality_predictor, args):
    st.title("Sleep Quality Predictor")
    
    # Initialize HealthAgent
    health_agent = HealthAgent()
    
    # Load health data
    # health_data = get_health_data(sleep_quality_predictor, args)

    predictor, predictions, analysis = mood_predictor_result()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Input", "Predictions", "Visualizations", 
        "Health Recommendations", "Chat Assistant"
    ])
    
    # Data Input Tab (Tab 1)
    with tab1:
        st.header("Your Health Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Samsung Health Data")
  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sensor path')
    parser.add_argument('--data_path', type=str, default='ifh_affect', help='Mode of operation: train or generate')
    parser.add_argument('--par_ID', type=int, default=1)
    parser.add_argument('--start_date', type=str, default='2020-03-20', help='date on which to analysis should start')
    parser.add_argument('--end_date', type=str, default='2020-03-26', help='date on which to analysis should end')

    args = parser.parse_args()

    # OURA_PATH = args.data_path + '/par_' + str(args.par_ID) + '/oura/'
    # SAMSUNG_PATH = args.data_path+ '/par_' + str( args.par_ID ) + '/samsung/'

    # print(OURA_PATH)
    # print(SAMSUNG_PATH)

    # args.start_date='2020-03-25'
    # args.end_date='2020-04-05'
    

    # Load sleepQualityPredictor class
    # sleep_quality_predictor = SleepQualityPredictor() 

    mood_predictor_obj = MoodPredictor()

    predictor, predictions, analysis = mood_predictor_result()


    #st.sidebar.markdown("# Data Loading Debug")
    
    # Check basic path existence
    # st.sidebar.markdown("## Path Check")
    # st.sidebar.write(f"Oura path exists: {os.path.exists(OURA_PATH)}")
    # st.sidebar.write(f"Samsung path exists: {os.path.exists(SAMSUNG_PATH)}")
    
    # Show directory contents
    st.sidebar.markdown("## Directory Contents")
    if os.path.exists(OURA_PATH):
        st.sidebar.write("Oura directory contents:")
        st.sidebar.write([f for f in os.listdir(OURA_PATH) if f.endswith('.csv')])
    
    if os.path.exists(SAMSUNG_PATH):
        st.sidebar.write("Samsung directory contents:")
        st.sidebar.write([f for f in os.listdir(SAMSUNG_PATH) if f.endswith('.csv')])
    
    create_streamlit_interface(mood_predictor_obj, args)
