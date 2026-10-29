"""
networks.py
PyTorch Deep Learning architectures for Air Traffic Forecasting:
1. AirTrafficLSTM: Stacked Long Short-Term Memory network with Dual Heads (Regression & Classification)
2. AirTrafficGRU: Gated Recurrent Unit network for comparative benchmarking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AirTrafficLSTM(nn.Module):
    """
    Multi-Task LSTM for:
    - Next-Hour Flight Traffic Volume Forecasting (Regression)
    - Next-Hour Congestion Level Prediction (Classification: Low=0, Medium=1, High=2)
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(AirTrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Shared representation layer
        self.fc_shared = nn.Linear(hidden_size, 32)
        
        # Head 1: Flight Volume Forecasting (Regression)
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Head 2: Congestion Level Classification (3 classes: Low, Medium, High)
        self.clf_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the representation of the last time step
        last_step = lstm_out[:, -1, :]
        normed = self.layer_norm(last_step)
        shared = F.relu(self.fc_shared(self.dropout(normed)))
        
        reg_out = self.reg_head(shared).squeeze(-1) # (batch_size,)
        clf_logits = self.clf_head(shared)          # (batch_size, 3)
        
        return reg_out, clf_logits


class AirTrafficGRU(nn.Module):
    """
    Multi-Task GRU (Gated Recurrent Unit) network for time-series benchmark comparison.
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(AirTrafficGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_shared = nn.Linear(hidden_size, 32)
        
        self.reg_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.clf_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        normed = self.layer_norm(last_step)
        shared = F.relu(self.fc_shared(self.dropout(normed)))
        
        reg_out = self.reg_head(shared).squeeze(-1)
        clf_logits = self.clf_head(shared)
        
        return reg_out, clf_logits
