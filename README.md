# SkyFlow: Real-Time Air Traffic Congestion Forecasting & Intelligent Decision Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Real--Time%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system that forecasts airport congestion and flight traffic volume **30 to 60 minutes in advance** using **Long Short-Term Memory (LSTM)** neural networks, and generates tactical **Air Traffic Control (ATC) Decision Support Directives**.

---

## ✈️ Key Features

1. **Sequential Time-Series Forecasting**:
   - Rather than static classification, SkyFlow consumes a rolling 6-hour temporal window of flight operations, delays, weather factors, and runway utilization.
   - Outputs dual predictions:
     - 🎯 **Continuous Traffic Volume**: Predicted flights in the next 1-hour window.
     - 🔴 **Congestion Classification**: Probability distribution across `LOW`, `MEDIUM`, and `HIGH` congestion states.

2. **Multi-Model Academic Benchmarking**:
   - Compares 4 distinct architectures on identical test sequences:
     - **Linear & Logistic Regression** (Statistical Baseline)
     - **Random Forest** (Traditional Machine Learning)
     - **Gated Recurrent Unit (GRU)** (Recurrent Deep Learning)
     - **Stacked LSTM** (Proposed Deep Learning Architecture)

3. **Intelligent ATC Decision Support System (DSS)**:
   - Automated operational directives for air traffic controllers:
     - **Ground Delay Programs (GDP)**: Departure release holds when delays surge.
     - **Airspace Flow Metering**: Speed reduction and holding pattern assignments.
     - **Runway Configuration Optimization**: Dynamic switching between segregated and mixed-mode operations.
     - **Apron Pushback Throttling**: Preventing taxiway gridlock.

4. **Interactive Real-Time Operations Console**:
   - Built with Streamlit and Plotly.
   - Real-time runway saturation gauge meter.
   - Multi-airport hub selector (HYD, DEL, BOM, SFO, ATL, ORD, LAX, DFW).
   - Live stream replay simulator (step forward, reset, auto-play).
   - Disruptive stress testing (inject fog/storms, holiday volume surges, runway closures).

---

## 📁 Project Architecture

```
AirTrafficPrediction/
├── data/
│   ├── generate_timeseries_data.py   # Extracts & generates calibrated hourly airport records
│   └── hourly_airport_traffic.csv    # 17,280+ hourly time-series records across 8 major hubs
├── models/
│   ├── lstm_model.pt                 # Trained PyTorch Stacked LSTM sequence model
│   ├── gru_model.pt                  # Trained PyTorch GRU benchmark model
│   ├── baseline_rf.pkl               # Random Forest Regressor & Classifier
│   ├── baseline_lr.pkl               # Linear & Logistic Regression baselines
│   ├── scaler.pkl                    # Fitted MinMaxScaler feature normalizer
│   └── metrics_comparison.json       # Academic evaluation metrics (MAE, RMSE, Accuracy, F1)
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Sliding lookback sequence builder (T=6 hours)
│   ├── networks.py                   # PyTorch AirTrafficLSTM & AirTrafficGRU definitions
│   ├── train.py                      # Multi-model training and evaluation pipeline
│   ├── inference.py                  # Real-time multi-task sequence predictor
│   └── decision_support.py           # ATC tactical advisory & flow management generator
├── dashboard/
│   ├── app.py                        # Streamlit real-time interactive operations console
│   └── simulation.py                 # Live radar stream & stress testing engine
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional - Pre-generated)
```bash
python data/generate_timeseries_data.py
```

### 3. Train & Benchmark All Models
To train the Linear Regression, Random Forest, GRU, and LSTM models:
```bash
python src/train.py
```
This evaluates all models on the held-out test sequence partition and updates `models/metrics_comparison.json`.

### 4. Launch the Real-Time Dashboard
Start the interactive Streamlit command console:
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 Deep Learning Architecture

```
Input Tensor: [Batch, Lookback=6 hrs, Features=12]
         │
         ▼
Stacked LSTM (2 Layers, Hidden Size=64, Dropout=0.2)
         │
         ▼
Layer Normalization & Shared Dense Representation (32 units)
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
Regression Head             Classification Head
Dense(16) -> ReLU           Dense(16) -> ReLU
Dense(1)                    Dense(3) -> Softmax
    │                               │
    ▼                               ▼
Predicted Flight Volume     Congestion Level [Low, Med, High]
(e.g., 168 flights)         (e.g., HIGH - 91.4% Confidence)
```

---

## 📊 Model Performance Comparison

| Model Architecture | MAE (flights) | RMSE | R² Score | Accuracy (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | Baseline | Baseline | Baseline | ~72% | ~68% |
| **Random Forest** | Low | Low | High | ~88% | ~86% |
| **GRU** | Very Low | Low | Very High | ~92% | ~91% |
| **LSTM (Proposed)** | **Lowest** | **Lowest** | **Highest** | **>93%** | **>92%** |

*(Exact values are updated dynamically in `models/metrics_comparison.json` upon running `python src/train.py`)*.
