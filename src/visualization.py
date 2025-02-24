import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.color_palette import ColorPaletteManager

# Initialize color palette manager
color_manager = ColorPaletteManager()

def create_emotion_radar(emotions_df, custom_colors=None):
    """Create a simple radar chart of emotions."""
    if emotions_df is None or emotions_df.empty:
        return None

    # Create the base figure
    fig = go.Figure()

    # Add the main trace
    fig.add_trace(go.Scatterpolar(
        r=emotions_df['score'],
        theta=emotions_df['label'],
        fill='toself',
        name='Emotions',
        line_color=color_manager.get_emotion_color('neutral'),
        fillcolor=color_manager.get_emotion_color('neutral')
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Emotion Distribution"
    )

    return fig

def create_intensity_gauge(intensity, emotion='neutral'):
    """Create a simple gauge chart for emotion intensity."""
    try:
        # Get color scheme
        color_scheme = color_manager.get_color_scheme(emotion)

        # Create base figure
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=intensity * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Emotional Intensity"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color_scheme['primary']},
                'steps': [
                    {'range': [0, 33], 'color': color_scheme['light']},
                    {'range': [33, 66], 'color': color_scheme['complementary'][0]},
                    {'range': [66, 100], 'color': color_scheme['dark']}
                ]
            }
        ))

        return fig
    except Exception as e:
        print(f"Error creating intensity gauge: {str(e)}")
        return None

def create_emotion_timeline(history_df, custom_colors=None):
    """Create a timeline of emotional changes."""
    if history_df is None or history_df.empty:
        return None

    # Create color map
    color_map = {emotion: color_manager.get_emotion_color(emotion)
                 for emotion in history_df['label'].unique()}
    if custom_colors:
        color_map.update(custom_colors)

    fig = px.line(history_df, 
                  x='timestamp', 
                  y='score',
                  color='label',
                  color_discrete_map=color_map,
                  title='Emotional Journey Over Time')

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Intensity",
        hovermode='x unified'
    )

    return fig

def create_emotional_arc_plot(arc_df, shifts=None, custom_colors=None):
    """Create an emotional arc visualization."""
    if arc_df is None or arc_df.empty:
        return None

    # Create base figure
    fig = go.Figure()

    # Add main arc line
    fig.add_trace(go.Scatter(
        x=arc_df['timestamp'],
        y=arc_df['current_intensity'],
        mode='lines',
        name='Emotional Intensity',
        line=dict(color=color_manager.get_emotion_color('neutral'))
    ))

    # Update layout
    fig.update_layout(
        title="Emotional Arc Analysis",
        xaxis_title="Time",
        yaxis_title="Emotional Intensity",
        hovermode='x unified'
    )

    return fig

def create_color_palette_preview(emotion):
    """Create a preview of the color palette for an emotion."""
    try:
        color_scheme = color_manager.get_color_scheme(emotion)

        # Create a simple bar chart showing the color scheme
        fig = go.Figure()

        # Prepare colors and labels
        colors = [
            color_scheme['primary'],
            *color_scheme['complementary'][:3],  # Take only first 3 complementary colors
            color_scheme['light'],
            color_scheme['dark']
        ]
        labels = [
            'Primary',
            'Complementary 1',
            'Complementary 2',
            'Complementary 3',
            'Light Variant',
            'Dark Variant'
        ]

        # Add bars for each color
        fig.add_trace(go.Bar(
            x=labels,
            y=[1] * len(colors),
            marker_color=colors,
            text=colors,
            textposition='auto',
        ))

        fig.update_layout(
            title=f"Color Palette Preview for {emotion.title()}",
            showlegend=False,
            yaxis={'showticklabels': False},
            plot_bgcolor='white'
        )

        return fig
    except Exception as e:
        print(f"Error creating color palette preview: {str(e)}")
        return None

def create_journal_emotion_calendar(entries_df):
    """Create a calendar heatmap of emotional intensities."""
    if entries_df is None or entries_df.empty:
        return None

    try:
        # Process dates and emotions for calendar view
        daily_emotions = entries_df.groupby(
            entries_df['timestamp'].dt.date
        )['emotional_intensity'].mean().reset_index()

        # Create heatmap
        fig = go.Figure(go.Heatmap(
            x=daily_emotions['timestamp'],
            y=['Intensity'],
            z=[daily_emotions['emotional_intensity'].values],
            colorscale=[
                [0, color_manager.get_emotion_color('neutral')],
                [0.5, color_manager.get_emotion_color('optimism')],
                [1, color_manager.get_emotion_color('joy')]
            ],
            showscale=True
        ))

        fig.update_layout(
            title='Emotional Intensity Calendar',
            xaxis_title='Date',
            yaxis_title='',
            height=200
        )

        return fig
    except Exception as e:
        print(f"Error creating journal calendar: {str(e)}")
        return None

def create_journal_emotion_summary(entries_df):
    """Create a summary visualization of emotional distribution in journal entries."""
    if entries_df is None or entries_df.empty:
        return None

    try:
        # Aggregate emotions
        emotions_list = []
        for _, row in entries_df.iterrows():
            emotions = row['emotions']
            if isinstance(emotions, dict):
                for emotion, score in emotions.items():
                    emotions_list.append({'emotion': emotion, 'score': score})

        if not emotions_list:
            return None

        # Convert to DataFrame and calculate means
        emotions_df = pd.DataFrame(emotions_list)
        emotion_summary = emotions_df.groupby('emotion')['score'].mean()

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=emotion_summary.index,
            y=emotion_summary.values,
            marker_color=[color_manager.get_emotion_color(emotion) 
                         for emotion in emotion_summary.index]
        ))

        fig.update_layout(
            title='Overall Emotional Distribution',
            xaxis_title='Emotion',
            yaxis_title='Average Intensity',
            showlegend=False,
            height=400
        )

        return fig
    except Exception as e:
        print(f"Error creating journal summary: {str(e)}")
        return None

def create_journal_timeline(entries_df):
    """Create an interactive timeline of journal entries."""
    if entries_df is None or entries_df.empty:
        return None

    try:
        fig = go.Figure()

        # Create scatter plot for entries
        fig.add_trace(go.Scatter(
            x=entries_df['timestamp'],
            y=entries_df['emotional_intensity'],
            mode='markers+lines',
            name='Emotional Intensity',
            marker=dict(
                size=12,
                color=[color_manager.get_emotion_color(emotion) 
                       for emotion in entries_df['primary_emotion']],
                symbol='circle'
            ),
            text=entries_df['text'],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Emotion: %{customdata[0]}<br>" +
                "Intensity: %{y:.2f}<br>" +
                "Time: %{x}<br>" +
                "<extra></extra>"
            ),
            customdata=entries_df[['primary_emotion']]
        ))

        fig.update_layout(
            title='Journal Entry Timeline',
            xaxis_title='Time',
            yaxis_title='Emotional Intensity',
            hovermode='closest',
            height=400
        )

        return fig
    except Exception as e:
        print(f"Error creating journal timeline: {str(e)}")
        return None