# src/pipeline.py

import pandas as pd
import json
from datetime import datetime
from src.load_data import load_all_csv
from src.feature_engineering import clean_and_engineer_features
from src.utils import calculate_soc, calculate_soh

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import os

# Directories
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)
METRICS_PATH = MODEL_DIR + "metrics.json"
PARQUET_PATH = "data/battery_data.parquet"

def load_dataset():
    try:
        print("📂 Trying to load Parquet data...")
        df = pd.read_parquet(PARQUET_PATH)
        print(f"✅ Loaded Parquet: {df.shape}")
    except Exception:
        print("⚠️ No Parquet found. Loading from CSVs...")
        df = load_all_csv()
        print(f"✅ Loaded CSV data: {df.shape}")
        print("💾 Saving Parquet for faster future loading...")
        df.to_parquet(PARQUET_PATH)
    return df

def train_all_models():
    print("📂 Loading and cleaning data...")
    df = load_dataset()
    df = clean_and_engineer_features(df)
    print(f"✅ Data ready for training: {df.shape}")

    # Features and target
    features = ["Voltage_measured", "Current_measured", "Power"]
    target = "Temperature_measured"

    X = df[features].fillna(0)
    y = df[target].fillna(0)

    print("📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define algorithms (SVM skipped)
    algorithms = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.05, n_jobs=-1, random_state=42)
    }

    metrics = {}
    best_rmse = float("inf")
    best_model_name = None
    best_model = None

    print("🚀 Training all models...")
    for name, model in algorithms.items():
        print(f"\nTraining {name}...")

        # Memory-efficient sampling for heavy models (except LightGBM)
        if name == "LightGBM":
            X_train_model = X_train
            y_train_model = y_train
        else:
            sample_size = min(100_000, len(X_train))  # sample 100k rows
            X_train_model = X_train.sample(n=sample_size, random_state=42)
            y_train_model = y_train.sample(n=sample_size, random_state=42)

        model.fit(X_train_model, y_train_model)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        print(f"{name} RMSE: {rmse:.4f}")
        metrics[name] = float(rmse)

        # Calculate SOC & SOH for test predictions
        soc = calculate_soc(preds)
        soh = calculate_soh(preds)
        metrics[f"{name}_SOC_mean"] = float(soc.mean())
        metrics[f"{name}_SOH_mean"] = float(soh.mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

    print(f"\n✅ Best model: {best_model_name} with RMSE={best_rmse:.4f}")

    # Save best model
    joblib.dump(best_model, MODEL_DIR + "best_model.pkl")
    print(f"💾 Best model saved at {MODEL_DIR}best_model.pkl")

    # Add additional info
    metrics["best_model"] = best_model_name
    metrics["best_rmse"] = float(best_rmse)
    metrics["n_train_samples"] = len(X_train)
    metrics["n_test_samples"] = len(X_test)
    metrics["n_features"] = len(features)
    metrics["train_time"] = str(datetime.now())

    # Save metrics to JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📄 Metrics saved at {METRICS_PATH}")
    print("✅ Pipeline finished.")

if __name__ == "__main__":
    train_all_models()
