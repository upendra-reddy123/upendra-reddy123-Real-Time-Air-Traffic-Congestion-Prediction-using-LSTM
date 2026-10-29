"""
dataset.py
Sequence generation and feature scaling for Time-Series LSTM/GRU and baseline models.
Uses a sliding lookback window (e.g. past 6 hours) to predict:
1. Total flights in the next hour (regression)
2. Congestion level in the next hour (classification: 0=Low, 1=Medium, 2=High)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

FEATURE_COLUMNS = [
    "arriving_flights",
    "departing_flights",
    "total_flights",
    "avg_arrival_delay",
    "avg_departure_delay",
    "weather_delay_factor",
    "taxi_out_avg",
    "runway_utilization",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical time encodings (hour of day, day of week)."""
    df = df.copy()
    
    # Cyclical hour encoding (0-23)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    
    # Cyclical day of week encoding (0-6)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    return df

def create_sliding_sequences(df: pd.DataFrame, lookback: int = 6):
    """
    Groups by airport and builds sliding lookback sequences.
    Returns:
    - X: (N, lookback, num_features)
    - y_reg: (N,) next hour flight count
    - y_clf: (N,) next hour congestion class
    """
    df_feat = engineer_features(df)
    
    all_X = []
    all_y_reg = []
    all_y_clf = []
    
    # Process each airport independently so sequences don't cross airport boundaries
    for airport, group in df_feat.groupby("airport"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feat_values = group[FEATURE_COLUMNS].values
        reg_values = group["total_flights"].values
        clf_values = group["congestion_level"].values
        
        n_rows = len(group)
        for i in range(n_rows - lookback):
            seq_x = feat_values[i : i + lookback]
            target_reg = reg_values[i + lookback]
            target_clf = clf_values[i + lookback]
            
            all_X.append(seq_x)
            all_y_reg.append(target_reg)
            all_y_clf.append(target_clf)
            
    return np.array(all_X, dtype=np.float32), np.array(all_y_reg, dtype=np.float32), np.array(all_y_clf, dtype=np.int64)

class AirTrafficDataModule:
    """
    Prepares scaled train, validation, and test sets for PyTorch and Scikit-Learn models.
    """
    def __init__(self, data_path: str, lookback: int = 6, test_ratio: float = 0.15, val_ratio: float = 0.15):
        self.data_path = data_path
        self.lookback = lookback
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.scaler = MinMaxScaler()
        
    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        X, y_reg, y_clf = create_sliding_sequences(df, lookback=self.lookback)
        
        N = len(X)
        test_size = int(N * self.test_ratio)
        val_size = int(N * self.val_ratio)
        train_size = N - val_size - test_size
        
        # Chronological train/val/test split
        X_train, y_reg_train, y_clf_train = X[:train_size], y_reg[:train_size], y_clf[:train_size]
        X_val, y_reg_val, y_clf_val = X[train_size:train_size + val_size], y_reg[train_size:train_size + val_size], y_clf[train_size:train_size + val_size]
        X_test, y_reg_test, y_clf_test = X[train_size + val_size:], y_reg[train_size + val_size:], y_clf[train_size + val_size:]
        
        # Scale features: fit scaler on flattened train data
        num_features = X_train.shape[2]
        X_train_flat = X_train.reshape(-1, num_features)
        self.scaler.fit(X_train_flat)
        
        def transform_seq(seq_arr):
            shape = seq_arr.shape
            flat = seq_arr.reshape(-1, num_features)
            scaled = self.scaler.transform(flat)
            return scaled.reshape(shape)
            
        X_train = transform_seq(X_train)
        X_val = transform_seq(X_val)
        X_test = transform_seq(X_test)
        
        return {
            "train": (X_train, y_reg_train, y_clf_train),
            "val": (X_val, y_reg_val, y_clf_val),
            "test": (X_test, y_reg_test, y_clf_test),
            "scaler": self.scaler,
            "feature_names": FEATURE_COLUMNS
        }

    def save_scaler(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler, 
                "features": FEATURE_COLUMNS,
                "lookback": self.lookback
            }, f)
