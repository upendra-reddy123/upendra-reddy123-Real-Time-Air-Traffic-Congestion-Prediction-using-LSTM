"""
train.py
Multi-Model Training and Comparative Evaluation Pipeline:
1. Linear Regression (Baseline)
2. Random Forest Regressor & Classifier (Traditional ML)
3. PyTorch GRU (Recurrent Deep Learning)
4. PyTorch LSTM (Proposed Deep Learning Model)

Evaluates all models on identical test sequences and exports:
- models/lstm_model.pt
- models/gru_model.pt
- models/baseline_rf.pkl
- models/baseline_lr.pkl
- models/scaler.pkl
- models/metrics_comparison.json
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

from src.dataset import AirTrafficDataModule
from src.networks import AirTrafficLSTM, AirTrafficGRU

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "hourly_airport_traffic.csv")

def evaluate_predictions(y_reg_true, y_reg_pred, y_clf_true, y_clf_pred):
    """Computes comprehensive regression and classification metrics."""
    mae = float(mean_absolute_error(y_reg_true, y_reg_pred))
    rmse = float(np.sqrt(mean_squared_error(y_reg_true, y_reg_pred)))
    r2 = float(r2_score(y_reg_true, y_reg_pred))
    
    acc = float(accuracy_score(y_clf_true, y_clf_pred))
    f1 = float(f1_score(y_clf_true, y_clf_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_clf_true, y_clf_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_clf_true, y_clf_pred, average="macro", zero_division=0))
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Accuracy": round(acc * 100, 2),
        "Macro_F1": round(f1 * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2)
    }

def train_pytorch_model(model_class, model_name, X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, X_test, y_reg_test, y_clf_test, epochs=15, batch_size=64):
    print(f"\n--- Training {model_name} (Deep Learning) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_size = X_train.shape[2]
    model = model_class(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_reg_train, dtype=torch.float32),
        torch.tensor(y_clf_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_reg_val, dtype=torch.float32),
        torch.tensor(y_clf_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_reg_test, dtype=torch.float32),
        torch.tensor(y_clf_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y_reg, batch_y_clf in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_clf = batch_y_clf.to(device)
            
            optimizer.zero_grad()
            pred_reg, pred_clf = model(batch_x)
            
            loss_reg = criterion_reg(pred_reg, batch_y_reg)
            loss_clf = criterion_clf(pred_clf, batch_y_clf)
            # Balance regression loss (scale ~50) and classification loss (scale ~1)
            loss = 0.02 * loss_reg + 1.2 * loss_clf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by_reg, by_clf in val_loader:
                bx, by_reg, by_clf = bx.to(device), by_reg.to(device), by_clf.to(device)
                preg, pclf = model(bx)
                lreg = criterion_reg(preg, by_reg)
                lclf = criterion_clf(pclf, by_clf)
                vloss = 0.02 * lreg + 1.2 * lclf
                val_loss += vloss.item() * len(bx)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    # Load best weights
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    
    # Test evaluation
    model.eval()
    all_preds_reg, all_preds_clf = [], []
    with torch.no_grad():
        for bx, _, _ in test_loader:
            bx = bx.to(device)
            preg, pclf = model(bx)
            all_preds_reg.extend(preg.cpu().numpy())
            pred_classes = torch.argmax(pclf, dim=1).cpu().numpy()
            all_preds_clf.extend(pred_classes)
            
    metrics = evaluate_predictions(y_reg_test, np.array(all_preds_reg), y_clf_test, np.array(all_preds_clf))
    print(f"{model_name} Test Results -> MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | Accuracy: {metrics['Accuracy']}% | F1: {metrics['Macro_F1']}%")
    
    return model, metrics

def run_training_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading data module and building sliding sequences...")
    dm = AirTrafficDataModule(DATA_PATH, lookback=6)
    data = dm.prepare_data()
    dm.save_scaler(os.path.join(MODELS_DIR, "scaler.pkl"))
    
    X_train, y_reg_train, y_clf_train = data["train"]
    X_val, y_reg_val, y_clf_val = data["val"]
    X_test, y_reg_test, y_clf_test = data["test"]
    
    print(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    results = {}
    
    # 1. Flattened features for Scikit-Learn Baselines
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    # --- MODEL 1: Linear Regression & Logistic Regression ---
    print("\n--- Training Model 1: Linear Regression Baseline ---")
    lr_reg = LinearRegression()
    lr_reg.fit(X_train_flat, y_reg_train)
    lr_pred_reg = lr_reg.predict(X_test_flat)
    
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train_flat, y_clf_train)
    lr_pred_clf = lr_clf.predict(X_test_flat)
    
    results["Linear Regression"] = evaluate_predictions(y_reg_test, lr_pred_reg, y_clf_test, lr_pred_clf)
    with open(os.path.join(MODELS_DIR, "baseline_lr.pkl"), "wb") as f:
        pickle.dump({"reg": lr_reg, "clf": lr_clf}, f)
    print(f"Linear Regression Results -> MAE: {results['Linear Regression']['MAE']} | Accuracy: {results['Linear Regression']['Accuracy']}%")
    
    # --- MODEL 2: Random Forest (Traditional ML) ---
    print("\n--- Training Model 2: Random Forest ---")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_flat, y_reg_train)
    rf_pred_reg = rf_reg.predict(X_test_flat)
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_flat, y_clf_train)
    rf_pred_clf = rf_clf.predict(X_test_flat)
    
    results["Random Forest"] = evaluate_predictions(y_reg_test, rf_pred_reg, y_clf_test, rf_pred_clf)
    with open(os.path.join(MODELS_DIR, "baseline_rf.pkl"), "wb") as f:
        pickle.dump({"reg": rf_reg, "clf": rf_clf}, f)
    print(f"Random Forest Results -> MAE: {results['Random Forest']['MAE']} | Accuracy: {results['Random Forest']['Accuracy']}%")
    
    # --- MODEL 3: PyTorch GRU (Deep Learning) ---
    gru_model, gru_metrics = train_pytorch_model(
        AirTrafficGRU, "GRU (Gated Recurrent Unit)",
        X_train, y_reg_train, y_clf_train,
        X_val, y_reg_val, y_clf_val,
        X_test, y_reg_test, y_clf_test,
        epochs=12, batch_size=64
    )
    results["GRU"] = gru_metrics
    torch.save(gru_model.state_dict(), os.path.join(MODELS_DIR, "gru_model.pt"))
    
    # --- MODEL 4: PyTorch LSTM (Proposed Deep Learning) ---
    lstm_model, lstm_metrics = train_pytorch_model(
        AirTrafficLSTM, "LSTM (Long Short-Term Memory)",
        X_train, y_reg_train, y_clf_train,
        X_val, y_reg_val, y_clf_val,
        X_test, y_reg_test, y_clf_test,
        epochs=15, batch_size=64
    )
    results["LSTM (Proposed)"] = lstm_metrics
    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
    
    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n========================================================")
    print("           MODEL PERFORMANCE COMPARISON TABLE           ")
    print("========================================================")
    print(f"{'Model':<20} | {'MAE':<6} | {'RMSE':<6} | {'R2':<6} | {'Accuracy':<9} | {'F1-Score':<8}")
    print("-" * 65)
    for model_name, m in results.items():
        print(f"{model_name:<20} | {m['MAE']:<6.2f} | {m['RMSE']:<6.2f} | {m['R2']:<6.4f} | {m['Accuracy']:>7.2f}% | {m['Macro_F1']:>6.2f}%")
    print("========================================================")
    print(f"All models and comparative metrics successfully saved to: {MODELS_DIR}")

if __name__ == "__main__":
    run_training_pipeline()
