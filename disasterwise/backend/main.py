import os
import shutil
import uuid
from datetime import datetime
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal imports
from database import init_db, store_prediction, get_history
from alert_engine import compute_alert
from model_wrapper import predict

app = FastAPI(title="DISASTERWISE API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, set to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Directory for storing uploaded images
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PredictionResponse(BaseModel):
    damage_percentage: float
    alert_level: str
    timestamp: str
    message: str

@app.get("/")
def home():
    return {"message": "Disasterwise Backend is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_damage(
    pre_image: UploadFile = File(...), 
    post_image: UploadFile = File(...)
):
    try:
        # Generate unique filenames
        request_id = str(uuid.uuid4())
        pre_path = os.path.join(UPLOAD_DIR, f"{request_id}_pre_{pre_image.filename}")
        post_path = os.path.join(UPLOAD_DIR, f"{request_id}_post_{post_image.filename}")

        # Save uploaded files
        with open(pre_path, "wb") as buffer:
            shutil.copyfileobj(pre_image.file, buffer)
        with open(post_path, "wb") as buffer:
            shutil.copyfileobj(post_image.file, buffer)

        # 1. CALL MODEL PREDICTION
        # Note: model_wrapper.predict is a placeholder the user can update with their actual weights.
        damage_mask, damage_percentage = predict(pre_path, post_path)

        # 2. COMPUTE ALERT
        # Uses alert_engine logic.
        alert_level, message, dsi = compute_alert(damage_percentage)

        # 3. STORE RESULT IN DATABASE
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store_prediction(
            pre_path, 
            post_path, 
            damage_percentage, 
            alert_level, 
            message
        )

        # 4. RETURN JSON RESPONSE
        return {
            "damage_percentage": round(damage_percentage, 2),
            "alert_level": alert_level,
            "timestamp": timestamp,
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

@app.get("/history")
async def fetch_history():
    try:
        return get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
