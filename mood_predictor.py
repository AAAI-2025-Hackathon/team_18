import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings


from pathlib import Path

warnings.filterwarnings('ignore')


class MoodPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def predict_mood(self, features):
        """Predict mood with proper feature preparation"""
        if self.model is None or self.feature_names is None:
            raise ValueError("Model needs to be trained first")
        
        # Prepare features the same way as during training
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_features 
                       if col not in ['date', 'mood', 'day_of_week']]
        
        # One-hot encode categorical variables
        categorical_features = features.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_features 
                          if col not in ['date', 'mood', 'activity_level']]
        
        if categorical_cols:
            features_encoded = pd.get_dummies(features[categorical_cols], 
                                            prefix=categorical_cols)
            X = pd.concat([features[numeric_cols], features_encoded], axis=1)
        else:
            X = features[numeric_cols]
        
        # Ensure all columns from training are present
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
            
        # Ensure columns are in the same order as during training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predictions = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': features['date'],
            'predicted_mood': predictions,
            'activity_level': features['activity_level'] if 'activity_level' in features else None
        })
        
        return results


    def extract_mood_features(self, samsung_data, oura_data):
        """Extract features relevant to mood prediction using correct data sources"""
        
        # Process Samsung IMU data for movement patterns
        imu_data = samsung_data['imu'].copy()
        imu_data['timestamp'] = pd.to_datetime(imu_data['timestamp'], unit='ms')
        imu_data['date'] = pd.to_datetime(imu_data['timestamp'].dt.date)  # Convert to datetime64
        
        # Calculate movement intensity
        imu_data['movement_intensity'] = np.sqrt(
            imu_data['accx']**2 + 
            imu_data['accy']**2 + 
            imu_data['accz']**2
        )
        
        # Calculate movement features
        movement_features = imu_data.groupby('date').agg({
            'movement_intensity': ['mean', 'std', 'max'],
            'gyrx': ['mean', 'std'],
            'gyry': ['mean', 'std'],
            'gyrz': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        movement_features.columns = ['date'] + [
            f'movement_{col[0]}_{col[1]}' if isinstance(col, tuple) else col 
            for col in movement_features.columns[1:]
        ]
        
        # Ensure date is datetime
        movement_features['date'] = pd.to_datetime(movement_features['date'])
        
        # Process Samsung pedometer data
        pedometer_data = samsung_data['pedometer'].copy()
        pedometer_data['timestamp'] = pd.to_datetime(pedometer_data['timestamp'], unit='ms')
        pedometer_data['date'] = pd.to_datetime(pedometer_data['timestamp'].dt.date)
        
        pedometer_features = pedometer_data.groupby('date').agg({
            'num_total_steps': 'max',
            'cal_burn_kcal': 'max',
            'last_speed_kmh': ['mean', 'max'],
            'last_state_class': lambda x: x.value_counts().index[0]
        }).reset_index()
        
        # Flatten pedometer column names and ensure date type
        pedometer_features.columns = ['date'] + [
            f'pedometer_{col[0]}_{col[1]}' if isinstance(col, tuple) else f'pedometer_{col}'
            for col in pedometer_features.columns[1:]
        ]
        pedometer_features['date'] = pd.to_datetime(pedometer_features['date'])
        
        # Process Samsung PPG data
        ppg_data = samsung_data['ppg'].copy()
        ppg_data['timestamp'] = pd.to_datetime(ppg_data['timestamp'], unit='ms')
        ppg_data['date'] = pd.to_datetime(ppg_data['timestamp'].dt.date)
        
        ppg_features = ppg_data.groupby('date').agg({
            'ppg': ['mean', 'std'],
            'hr': lambda x: np.mean(x[x > 0])
        }).reset_index()
        
        # Flatten PPG column names and ensure date type
        ppg_features.columns = ['date'] + [
            f'ppg_{col[0]}_{col[1]}' if isinstance(col, tuple) else f'ppg_{col}'
            for col in ppg_features.columns[1:]
        ]
        ppg_features['date'] = pd.to_datetime(ppg_features['date'])
        
        # Process Oura activity data
        activity_features = oura_data['activity'].copy()
        activity_features['date'] = pd.to_datetime(activity_features['date'])
        activity_cols = [
            'date', 'score', 'score_stay_active', 'score_move_every_hour',
            'cal_active', 'cal_total', 'daily_movement', 'steps',
            'inactive', 'low', 'medium', 'high', 'average_met'
        ]
        activity_features = activity_features[activity_cols].copy()
        activity_features.columns = ['date'] + [f'activity_{col}' for col in activity_features.columns[1:]]
        
        # Process Oura sleep data
        sleep_features = oura_data['sleep'].copy()
        sleep_features['date'] = pd.to_datetime(sleep_features['date'])
        sleep_cols = [
            'date', 'score', 'score_deep', 'score_rem', 'score_efficiency',
            'hr_average', 'hr_lowest', 'rmssd', 'temperature_delta'
        ]
        sleep_features = sleep_features[sleep_cols].copy()
        sleep_features.columns = ['date'] + [f'sleep_{col}' for col in sleep_features.columns[1:]]
        
        # Process Oura readiness data
        readiness_features = oura_data['readiness'].copy()
        readiness_features['date'] = pd.to_datetime(readiness_features['date'])
        readiness_cols = [
            'date', 'score', 'score_activity_balance', 'score_hrv_balance',
            'score_recovery_index', 'score_sleep_balance'
        ]
        readiness_features = readiness_features[readiness_cols].copy()
        readiness_features.columns = ['date'] + [f'readiness_{col}' for col in readiness_features.columns[1:]]
        
        # Debug prints to check date types
        print("Date types before merging:")
        print("Movement features date type:", movement_features['date'].dtype)
        print("Pedometer features date type:", pedometer_features['date'].dtype)
        print("PPG features date type:", ppg_features['date'].dtype)
        print("Activity features date type:", activity_features['date'].dtype)
        print("Sleep features date type:", sleep_features['date'].dtype)
        print("Readiness features date type:", readiness_features['date'].dtype)
        
        # Combine all features
        features = movement_features.copy()
        
        # List of DataFrames to merge
        dfs_to_merge = [
            pedometer_features,
            ppg_features,
            activity_features,
            sleep_features,
            readiness_features
        ]
        
        # Merge all features
        for df in dfs_to_merge:
            features = features.merge(df, on='date', how='outer')
        
        # Create derived features using Oura activity data
        features['activity_level'] = pd.qcut(features['activity_daily_movement'], 
                                        q=5, 
                                        labels=['very_low', 'low', 'moderate', 'high', 'very_high'])
        
        features['activity_ratio'] = (features['activity_medium'] + features['activity_high']) / \
                                (features['activity_inactive'] + features['activity_low'] + 1e-6)
        
        # Add time-based features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        return features


    def analyze_mood_factors(self, features, predictions):
        """Analyze which factors contribute most to mood prediction"""
        # Get only numeric columns
        numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_columns 
                    if col not in ['date', 'mood', 'day_of_week']]
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,  # Use stored feature names from training
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate correlations only for numeric features
        correlations = {}
        for col in feature_cols:
            try:
                # Handle potential NaN values
                valid_mask = ~(features[col].isna() | predictions['predicted_mood'].isna())
                if valid_mask.any():  # Only calculate if we have valid data
                    correlation = np.corrcoef(
                        features[col][valid_mask],
                        predictions['predicted_mood'][valid_mask]
                    )[0,1]
                    correlations[col] = correlation
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {col}: {e}")
        
        correlation_df = pd.DataFrame({
            'feature': correlations.keys(),
            'correlation': correlations.values()
        }).sort_values('correlation', ascending=False)
        
        return {
            'feature_importance': importance,
            'correlations': correlation_df
        }





def create_mood_predictor(samsung_data, oura_data):
    """Create and return a mood predictor instance with proper handling of categorical variables"""
    predictor = MoodPredictor()
    
    # Extract features
    features = predictor.extract_mood_features(samsung_data, oura_data)
    
    # For demonstration, create synthetic mood labels
    np.random.seed(42)
    features['mood'] = np.random.randint(1, 6, size=len(features))
    
    # Identify numeric and categorical columns
    numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = features.select_dtypes(include=['object', 'category']).columns
    
    # Print column types for debugging
    print("\nNumeric columns:", numeric_columns.tolist())
    print("\nCategorical columns:", categorical_columns.tolist())
    
    # Remove date and target columns from feature lists
    feature_cols = [col for col in numeric_columns 
                   if col not in ['date', 'mood', 'day_of_week']]
    
    # One-hot encode categorical variables if needed
    categorical_features = [col for col in categorical_columns 
                          if col not in ['date', 'mood', 'activity_level']]
    
    if categorical_features:
        features_encoded = pd.get_dummies(features[categorical_features], 
                                        prefix=categorical_features)
        X = pd.concat([features[feature_cols], features_encoded], axis=1)
    else:
        X = features[feature_cols]
    
    y = features['mood']
    
    # Print final feature set for verification
    print("\nFinal features used:", X.columns.tolist())
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale only numeric features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train model
    predictor.model = RandomForestClassifier(n_estimators=100, random_state=42)
    predictor.model.fit(X_train_scaled, y_train)
    
    # Store feature names for later use
    predictor.feature_names = X.columns.tolist()
    
    return predictor, features



def load_sensor_data(data_directory):
    """Load all sensor data from specified directory"""
    samsung_data_directory = os.path.join(data_directory, 'par_1/samsung')
    oura_data_directory = os.path.join(data_directory, 'par_1/oura' )

    pd.read_csv(Path(samsung_data_directory) / 'awake_times.csv')
    
    try:
        # Load Samsung sensor data
        samsung_data = {
            'awake_times': pd.read_csv(Path(samsung_data_directory) / 'awake_times.csv'),
            'hrv_1min': pd.read_csv(Path(samsung_data_directory) / 'hrv_1min.csv'),
            'hrv_5min': pd.read_csv(Path(samsung_data_directory) / 'hrv_5min.csv'),
            'imu': pd.read_csv(Path(samsung_data_directory) / 'imu.csv'),
            'pedometer': pd.read_csv(Path(samsung_data_directory) / 'pedometer.csv'),
            'ppg': pd.read_csv(Path(samsung_data_directory) / 'ppg.csv'),
            'pressure': pd.read_csv(Path(samsung_data_directory) / 'pressure.csv'),
        }
        
        # Load Oura data
        oura_data = {
            'activity': pd.read_csv(Path(oura_data_directory) / 'activity.csv'),
            'activity_level': pd.read_csv(Path(oura_data_directory) / 'activity_level.csv'),
            'heart_rate': pd.read_csv(Path(oura_data_directory) / 'heart_rate.csv'),
            'readiness': pd.read_csv(Path(oura_data_directory) / 'readiness.csv'),
            'sleep': pd.read_csv(Path(oura_data_directory) / 'sleep.csv'),
            'sleep_hypnogram': pd.read_csv(Path(oura_data_directory) / 'sleep_hypnogram.csv')
        }
        
        # Initial data validation
        validate_data(samsung_data, oura_data)
        
        return samsung_data, oura_data
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None
    except pd.errors.EmptyDataError as e:
        print(f"Error: Empty CSV file found: {e}")
        return None, None

def validate_data(samsung_data, oura_data):
    """Validate loaded data for completeness and basic format"""
    # Check Samsung data
    for name, df in samsung_data.items():
        if 'timestamp' not in df.columns and 'timestamp_start' not in df.columns:
            print(f"Warning: No timestamp column in {name}")
        print(f"Loaded {name}: {len(df)} rows")
            
    # Check Oura data
    for name, df in oura_data.items():
        if 'date' not in df.columns and 'timestamp' not in df.columns:
            print(f"Warning: No date/timestamp column in {name}")
        print(f"Loaded {name}: {len(df)} rows")

def prepare_data_for_training(samsung_data, oura_data):
    """Prepare loaded data for mood prediction training"""
    
    # Extract relevant features from both Samsung and Oura data
    predictor = MoodPredictor()
    
    # Prepare features from Samsung data
    features = predictor.extract_mood_features(
        imu_data=samsung_data['imu'],
        pedometer_data=samsung_data['pedometer'],
        ppg_data=samsung_data['ppg'],
        hrv_data=samsung_data['hrv_1min'],
        sleep_data=oura_data['sleep_hypnogram']
    )
    
    # Add additional features from Oura data
    features = enrich_features_with_oura_data(features, oura_data)
    
    return features

def enrich_features_with_oura_data(features, oura_data):
    """Add Oura-specific features to the feature set"""
    
    # Convert date column to datetime if it's not already
    features['date'] = pd.to_datetime(features['date'])
    
    # Add sleep scores
    sleep_data = oura_data['sleep'].copy()
    sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    sleep_features = sleep_data[['date', 'score', 'score_deep', 'score_rem', 
                                'score_efficiency', 'hr_average', 'hr_lowest', 'rmssd']]
    features = features.merge(sleep_features, on='date', how='left', 
                            suffixes=('', '_sleep'))
    
    # Add readiness scores
    readiness_data = oura_data['readiness'].copy()
    readiness_data['date'] = pd.to_datetime(readiness_data['date'])
    readiness_features = readiness_data[['date', 'score', 'score_activity_balance',
                                       'score_hrv_balance', 'score_recovery_index']]
    features = features.merge(readiness_features, on='date', how='left',
                            suffixes=('', '_readiness'))
    
    # Add activity scores
    activity_data = oura_data['activity'].copy()
    activity_data['date'] = pd.to_datetime(activity_data['date'])
    activity_features = activity_data[['date', 'score', 'score_stay_active',
                                     'daily_movement', 'average_met']]
    features = features.merge(activity_features, on='date', how='left',
                            suffixes=('', '_activity'))
    
    return features

# def train_mood_predictor(data_directory):
#     """Main function to load data and train the mood predictor"""
    
#     print("Loading sensor data...")
#     samsung_data, oura_data = load_sensor_data(data_directory)
    
#     if samsung_data is None or oura_data is None:
#         return None
    
#     print("\nPreparing features for training...")
#     features = prepare_data_for_training(samsung_data, oura_data)
    
#     print("\nTraining mood predictor...")
#     predictor = create_mood_predictor(
#         samsung_data['imu'],
#         samsung_data['pedometer'],
#         samsung_data['ppg'],
#         samsung_data['hrv_1min'],
#         oura_data['sleep_hypnogram']
#     )
    
#     return predictor, features

# # Usage example
def train_mood_predictor(data_directory):
    """Main function to load data and train the mood predictor"""
    # Load data using the data loader
    samsung_data, oura_data = load_sensor_data(data_directory)
    
    if samsung_data is None or oura_data is None:
        return None
    
    # Create and train the predictor
    predictor, features = create_mood_predictor(samsung_data, oura_data)
    
    return predictor, features

# Usage example
def mood_predictor_result():
    # Specify your data directory
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(current_dir, 'ifh_affect_short')

    data_dir = "/Users/samanehmovassaghi/Downloads/EmoSenseAI/src/ifh_affect_short"
    
    # Train the predictor
    predictor, features = train_mood_predictor(data_dir)
    
    if predictor is not None:
        # Make predictions
        predictions = predictor.predict_mood(features)
        
        # Analyze factors
        analysis = predictor.analyze_mood_factors(features, predictions)
        
        # Print results
        print("\nPrediction Results:")
        print(predictions.head())
        
        print("\nTop Mood Factors:")
        print(analysis['feature_importance'].head())
        
        return predictor, predictions, analysis
    
    return None, None, None

if __name__ == "__main__":
    predictor, predictions, analysis = mood_predictor_result()


