# Backend/app/services/crop_service.py
import os
import pickle
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Moves two levels up
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
logger.info(f"Loading model from: {MODEL_PATH}")

# Load model at startup
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully!")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Feature names expected by the trained model
FEATURE_NAMES = ["N", "P", "K_log", "temperature", "humidity", "ph", "rainfall"]

def predict_top_crops(input_data):
    """
    Predict top 4 crops based on soil and weather data.
    
    input_data: list of 7 features [N, P, K, Temperature, Humidity, pH, Rainfall]
    Returns: list of dictionaries with crop name and probability
    """
    # Input validation
    if not isinstance(input_data, (list, tuple, np.ndarray)):
        raise ValueError("Input data must be a list, tuple, or numpy array")
    if len(input_data) != 7:
        raise ValueError(
            "Input must have 7 features: [N, P, K, Temperature, Humidity, pH, Rainfall]"
        )

    # Extract features
    N, P, K, temperature, humidity, ph, rainfall = input_data

    # Transform K -> K_log
    K_log = np.log(K) if K > 0 else 0  # Avoid log(0)

    # Prepare DataFrame with correct feature names
    data = pd.DataFrame([[N, P, K_log, temperature, humidity, ph, rainfall]], columns=FEATURE_NAMES)

    # Predict probabilities
    try:
        probabilities = model.predict_proba(data)
        classes = model.classes_
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    # Sort crops by probability and take top 4
    top_indices = np.argsort(probabilities[0])[::-1][:4]
    top_crops = [
        {"crop": classes[i], "probability": round(float(probabilities[0][i]), 4)}
        for i in top_indices
    ]

    return top_crops

# Test code
if __name__ == "__main__":
    test_input = [90, 42, 43, 20.87, 82.00, 6.50, 202.93]
    logger.info("Testing crop recommendation service...")
    try:
        results = predict_top_crops(test_input)
        for crop in results:
            logger.info(f"Crop: {crop['crop']}, Probability: {crop['probability']}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
