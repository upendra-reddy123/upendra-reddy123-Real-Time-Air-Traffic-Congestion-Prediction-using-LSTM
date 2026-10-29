"""
test_inference.py
Unit and integration test for multi-model inference and decision support.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import AirTrafficPredictor
from dashboard.simulation import TrafficSimulator

def test_pipeline():
    print("Testing Air Traffic Inference across all architectures...")
    
    sim = TrafficSimulator("HYD")
    df_history = sim.get_recent_history(lookback=6)
    
    models = ["LSTM", "GRU", "Random Forest", "Linear Regression"]
    for m in models:
        predictor = AirTrafficPredictor(model_type=m)
        res = predictor.predict_sequence(df_history, airport="HYD", capacity=55)
        print(f"\n[Model: {m}]")
        print(f"  Current Flights: {res['current_flights']} | Predicted Next-Hour: {res['predicted_flights']}")
        print(f"  Congestion Level: {res['congestion_label']} ({res['confidence']}%)")
        print(f"  Probabilities: {res['class_probabilities']}")
        print(f"  Runway Utilization: {res['predicted_runway_utilization']}%")
        print(f"  Top Directive: {res['atc_advisories']['flow_actions'][0]}")

if __name__ == "__main__":
    test_pipeline()
