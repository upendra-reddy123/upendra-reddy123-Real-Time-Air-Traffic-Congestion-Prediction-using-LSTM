"""
inference.py
Real-time sequence inference engine.
Takes rolling historical time-series sequences (e.g., past 6 hours), runs model inference,
and returns dual predictions:
1. Expected traffic volume (next hour)
2. Congestion classification + confidence score
3. Operational ATC advisories
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.networks import AirTrafficLSTM, AirTrafficGRU
from src.dataset import FEATURE_COLUMNS, engineer_features
from src.decision_support import generate_atc_advisories, CONGESTION_LABELS

MODELS_DIR = os.path.join(BASE_DIR, "models")

class AirTrafficPredictor:
    def __init__(self, model_type: str = "LSTM"):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.features = FEATURE_COLUMNS
        self.lookback = 6
        self.model = None
        self.rf_model = None
        self.lr_model = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                saved = pickle.load(f)
                self.scaler = saved["scaler"]
                self.features = saved.get("features", FEATURE_COLUMNS)
                self.lookback = saved.get("lookback", 6)
                
        # Load Deep Learning Models
        if self.model_type == "LSTM":
            lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
            if os.path.exists(lstm_path):
                self.model = AirTrafficLSTM(input_size=len(self.features)).to(self.device)
                self.model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
                self.model.eval()
        elif self.model_type == "GRU":
            gru_path = os.path.join(MODELS_DIR, "gru_model.pt")
            if os.path.exists(gru_path):
                self.model = AirTrafficGRU(input_size=len(self.features)).to(self.device)
                self.model.load_state_dict(torch.load(gru_path, map_location=self.device, weights_only=True))
                self.model.eval()
        elif self.model_type == "Random Forest":
            rf_path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, "rb") as f:
                    self.rf_model = pickle.load(f)
        elif self.model_type == "Linear Regression":
            lr_path = os.path.join(MODELS_DIR, "baseline_lr.pkl")
            if os.path.exists(lr_path):
                with open(lr_path, "rb") as f:
                    self.lr_model = pickle.load(f)

    def predict_sequence(self, sequence_df: pd.DataFrame, airport: str, capacity: int = 60) -> Dict[str, Any]:
        """
        Accepts DataFrame with the past lookback hours (e.g. 6 rows) and generates real-time forecast.
        """
        if len(sequence_df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} historical rows, received {len(sequence_df)}")
            
        df_tail = sequence_df.tail(self.lookback).copy()
        df_tail = engineer_features(df_tail)
        
        # Raw features matrix: (lookback, num_features)
        feat_matrix = df_tail[self.features].values
        
        # Scale features
        if self.scaler is not None:
            scaled_feat = self.scaler.transform(feat_matrix)
        else:
            scaled_feat = feat_matrix
            
        current_row = df_tail.iloc[-1]
        current_flights = int(current_row.get("total_flights", 0))
        current_delay = float(current_row.get("avg_departure_delay", 0.0))
        current_util = float(current_row.get("runway_utilization", 0.0))
        weather_delay = float(current_row.get("weather_delay_factor", 0.0))

        # Prediction according to selected model architecture
        if self.model_type in ["LSTM", "GRU"] and self.model is not None:
            input_tensor = torch.tensor(scaled_feat, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, seq_len, feat)
            with torch.no_grad():
                pred_reg_t, pred_clf_t = self.model(input_tensor)
                pred_flights = max(0, int(round(pred_reg_t.item())))
                
                probs = F.softmax(pred_clf_t, dim=1).squeeze(0).cpu().numpy()
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class]) * 100.0
                
        elif self.model_type == "Random Forest" and self.rf_model is not None:
            flat_feat = scaled_feat.reshape(1, -1)
            pred_flights = max(0, int(round(self.rf_model["reg"].predict(flat_feat)[0])))
            probs = self.rf_model["clf"].predict_proba(flat_feat)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100.0
            
        elif self.model_type == "Linear Regression" and self.lr_model is not None:
            flat_feat = scaled_feat.reshape(1, -1)
            pred_flights = max(0, int(round(self.lr_model["reg"].predict(flat_feat)[0])))
            probs = self.lr_model["clf"].predict_proba(flat_feat)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100.0
            
        else:
            # Fallback heuristic if models are building
            pred_flights = current_flights
            pred_class = int(current_row.get("congestion_level", 0))
            probs = [0.33, 0.33, 0.33]
            confidence = 80.0

        # Calculate predicted runway utilization
        pred_util = min(100.0, round((pred_flights / capacity) * 100.0, 1))

        # Generate ATC Decision Support direct advisories
        atc_advisories = generate_atc_advisories(
            airport=airport,
            current_flights=current_flights,
            predicted_flights=pred_flights,
            congestion_level=pred_class,
            confidence_pct=round(confidence, 1),
            runway_utilization=pred_util,
            avg_delay=current_delay,
            weather_delay=weather_delay,
            capacity=capacity
        )

        return {
            "model_used": self.model_type,
            "airport": airport,
            "current_flights": current_flights,
            "current_delay_min": current_delay,
            "current_runway_utilization": current_util,
            "predicted_flights": pred_flights,
            "predicted_runway_utilization": pred_util,
            "congestion_level": pred_class,
            "congestion_label": CONGESTION_LABELS[pred_class],
            "confidence": round(confidence, 1),
            "class_probabilities": {
                "Low": round(float(probs[0]) * 100, 1),
                "Medium": round(float(probs[1]) * 100, 1),
                "High": round(float(probs[2]) * 100, 1)
            },
            "atc_advisories": atc_advisories
        }
