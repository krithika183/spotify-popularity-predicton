from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and the original dataset for feature median calculations
try:
    model = joblib.load('popularity_predictor_final.pkl')
    # Load the original dataset to calculate medians for imputation if needed
    df_original = pd.read_csv('Spotify_data.csv')

    # Pre-calculate medians for numerical features from the original dataset
    # This ensures consistency with how the model was trained
    # Note: 'Duration (ms)' has a space, 'Release Date' is for year/month extraction
    feature_columns_for_medians = [
        'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
        'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
        'Valence', 'Tempo', 'Duration (ms)'
    ]
    # Ensure 'Explicit' and 'Release Date' are handled for median calculation context
    if 'Explicit' in df_original.columns:
        df_original['Explicit'] = df_original['Explicit'].astype(int)
    if 'Release Date' in df_original.columns:
        df_original['Release Date'] = pd.to_datetime(df_original['Release Date'], errors='coerce')
        df_original['Release_Year'] = df_original['Release Date'].dt.year
        df_original['Release_Month'] = df_original['Release Date'].dt.month
        feature_columns_for_medians.extend(['Release_Year', 'Release_Month'])

    # Calculate medians for all relevant numerical features
    feature_medians = {}
    for col in feature_columns_for_medians:
        if col in df_original.columns and pd.api.types.is_numeric_dtype(df_original[col]):
            feature_medians[col] = df_original[col].median()
        elif col == 'Explicit': # Explicit is handled separately
            feature_medians[col] = 0 # Default explicit to 0 if not found in data
        elif col == 'Release_Year':
            feature_medians[col] = df_original['Release_Year'].median() if 'Release_Year' in df_original.columns else 2000
        elif col == 'Release_Month':
            feature_medians[col] = df_original['Release_Month'].median() if 'Release_Month' in df_original.columns else 1

    print("Model and original data loaded successfully.")
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None
    df_original = None
    feature_medians = {}

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles song popularity prediction requests.
    Expects a JSON payload with song features.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    data = request.get_json(force=True)

    # Define the expected features in the order the model expects them
    # This order MUST match the order of features used during model training
    expected_features = [
        'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
        'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
        'Valence', 'Tempo', 'Duration (ms)',
        'Explicit', 'Release_Year', 'Release_Month'
    ]

    processed_data = {}
    for feature in expected_features:
        value = data.get(feature)

        if value is None:
            # Impute missing values with pre-calculated medians
            processed_data[feature] = feature_medians.get(feature, 0) # Default to 0 if median not found
            print(f"Warning: Feature '{feature}' not provided. Imputing with median: {processed_data[feature]}")
        else:
            try:
                if feature == 'Explicit':
                    # Convert 'True'/'False' strings to 1/0
                    processed_data[feature] = 1 if str(value).lower() == 'true' else 0
                elif feature == 'Release Date': # This will be handled for year/month
                    # We expect Release_Year and Release_Month directly from the frontend
                    pass
                elif feature in ['Release_Year', 'Release_Month']:
                    processed_data[feature] = int(value)
                else:
                    processed_data[feature] = float(value)
            except ValueError:
                # Fallback to median if conversion fails
                processed_data[feature] = feature_medians.get(feature, 0)
                print(f"Warning: Could not convert '{feature}' value '{value}'. Imputing with median: {processed_data[feature]}")

    # Handle Release Date to Release_Year and Release_Month conversion if date string is sent
    # The frontend is designed to send Release_Year and Release_Month directly,
    # but this is a fallback/robustness check.
    if 'release_date' in data and 'Release_Year' not in processed_data:
        try:
            release_date_str = data.get('release_date')
            dt_object = datetime.strptime(release_date_str, '%Y-%m-%d')
            processed_data['Release_Year'] = dt_object.year
            processed_data['Release_Month'] = dt_object.month
        except (ValueError, TypeError):
            print("Warning: Invalid 'release_date' format. Using median for year/month.")
            processed_data['Release_Year'] = feature_medians.get('Release_Year', 2000)
            processed_data['Release_Month'] = feature_medians.get('Release_Month', 1)

    # Create a DataFrame for prediction, ensuring column order
    input_df = pd.DataFrame([processed_data], columns=expected_features)

    try:
        prediction = model.predict(input_df)[0]
        # Popularity is typically an integer score
        return jsonify({'popularity': int(round(prediction))})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False in production