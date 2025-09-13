# Backend/app/services/crop_service.py
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
print("Loading model from:", MODEL_PATH)

# Load model at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_top_crops(input_data):
    """
    Predict top 4 crops based on soil and weather data.
    input_data: [N, P, K, Temperature, Humidity, pH, Rainfall]
    """
    # Convert input into numpy array
    data = np.array([input_data])

    # Get prediction probabilities
    probabilities = model.predict_proba(data)

    # Get crop labels
    classes = model.classes_

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
    print("Testing crop recommendation service...\n")
    results = predict_top_crops(test_input)
    for crop in results:
        print(f"Crop: {crop['crop']}, Probability: {crop['probability']}")
