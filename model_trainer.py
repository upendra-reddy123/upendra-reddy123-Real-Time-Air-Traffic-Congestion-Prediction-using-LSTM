import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    print("Loading dataset...")
    data_path = os.path.join("data", "air_traffic_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # Features to use
    categorical_cols = ['Terminal', 'GEO Region', 'Activity Type Code', 'Month']
    numeric_cols = ['Year']
    target = 'Passenger Count'
    
    print("Pre-processing data...")
    # Drop rows with missing values in our features
    df = df.dropna(subset=categorical_cols + numeric_cols + [target])
    
    # Encode categories
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForest model... This might take a moment.")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")

    print("Saving model and encoders...")
    with open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders}, f)
    print("Model saved to model.pkl successfully!")

if __name__ == "__main__":
    train_model()
