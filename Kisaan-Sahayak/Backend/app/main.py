# Backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
# Use absolute import
from app.model.services.crop_service import predict_top_crops





# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kisaan Sahayak Crop Recommendation API",
    description="Predicts top 4 crops based on soil and weather features",
    version="1.0"
)

# Request model
class CropRequest(BaseModel):
    N: float
    P: float
    K: float  # raw K
    Temperature: float
    Humidity: float
    pH: float
    Rainfall: float

# Response model
class CropResponse(BaseModel):
    crop: str
    probability: float

@app.post("/predict", response_model=list[CropResponse])
def predict_crops(request: CropRequest):
    try:
        # Prepare input list for crop_service
        input_data = [
            request.N,
            request.P,
            request.K,  # raw K, service will convert to K_log
            request.Temperature,
            request.Humidity,
            request.pH,
            request.Rainfall
        ]

        results = predict_top_crops(input_data)
        return results

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
