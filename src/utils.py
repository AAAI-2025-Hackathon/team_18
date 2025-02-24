import pandas as pd
from datetime import datetime, timedelta

def init_session_state(st):
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = pd.DataFrame(
            columns=['timestamp', 'text', 'label', 'score']
        )

def update_emotion_history(st, emotions_df):
    if emotions_df is not None:
        st.session_state.emotion_history = pd.concat(
            [st.session_state.emotion_history, emotions_df],
            ignore_index=True
        )

def get_recent_history(st, hours=24):
    if st.session_state.emotion_history.empty:
        return None
        
    cutoff_time = datetime.now() - timedelta(hours=hours)
    return st.session_state.emotion_history[
        st.session_state.emotion_history['timestamp'] > cutoff_time
    ]
