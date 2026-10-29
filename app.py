from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import pandas as pd
import os

app = FastAPI(title="Air Traffic Congestion Prediction")

# Attempt to load model
model_data = None
model_path = "model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
else:
    print("Warning: model.pkl not found. Please train the model first.")

class PredictionRequest(BaseModel):
    year: int
    month: str
    terminal: str
    region: str
    activity: str

def get_congestion_level(passengers: float) -> str:
    if passengers < 20000:
        return "Low"
    elif passengers < 100000:
        return "Moderate"
    else:
        return "High"

@app.post("/api/predict")
def predict_congestion(req: PredictionRequest):
    if not model_data:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")
    
    model = model_data["model"]
    encoders = model_data["encoders"]

    try:
        # Encode inputs safely
        def encode_value(encoder, value):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                # Fallback if unknown
                return 0

        terminal_enc = encode_value(encoders["Terminal"], req.terminal)
        region_enc = encode_value(encoders["GEO Region"], req.region)
        activity_enc = encode_value(encoders["Activity Type Code"], req.activity)
        month_enc = encode_value(encoders["Month"], req.month)

        # Create input df
        x_input = pd.DataFrame([{
            'Year': req.year,
            'Terminal': terminal_enc,
            'GEO Region': region_enc,
            'Activity Type Code': activity_enc,
            'Month': month_enc
        }])

        pred_passengers = model.predict(x_input)[0]
        pred_passengers = max(0, int(pred_passengers))
        congestion = get_congestion_level(pred_passengers)

        return {
            "prediction": pred_passengers,
            "congestion_level": congestion
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Deep Learning (LSTM & Real-Time Decision Support) Endpoint ---
class LSTMPredictionRequest(BaseModel):
    airport: str = "HYD"
    model_type: str = "LSTM" # "LSTM", "GRU", "Random Forest", "Linear Regression"

@app.post("/api/v2/predict-congestion-lstm")
def predict_lstm_congestion(req: LSTMPredictionRequest):
    try:
        from src.inference import AirTrafficPredictor
        from dashboard.simulation import TrafficSimulator
        
        sim = TrafficSimulator(req.airport)
        recent_df = sim.get_recent_history(lookback=6)
        
        capacities = {"HYD": 55, "DEL": 90, "BOM": 65, "SFO": 70, "ATL": 120, "ORD": 100, "LAX": 95}
        cap = capacities.get(req.airport, 60)
        
        predictor = AirTrafficPredictor(model_type=req.model_type)
        res = predictor.predict_sequence(recent_df, airport=req.airport, capacity=cap)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configure static files delivery
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")
