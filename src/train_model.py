# src/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import json
from datetime import datetime
from src.load_data import load_all_csv
from src.feature_engineering import clean_and_engineer_features

# Paths
MODEL_DIR = "models/"
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
PARQUET_PATH = "data/battery_data.parquet"

# Make sure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset():
    """Load dataset from Parquet or CSV files."""
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

    # 🧠 Reduce dataset size automatically if too large
    MAX_SAMPLES = 300_000  # adjust if you have more memory
    
    
    
    # MAX_SAMPLES = 800_000  # or 1_000_000 if you have >16GB RAM




    if len(df) > MAX_SAMPLES:
        print(f"📉 Sampling {MAX_SAMPLES} rows from {len(df)} to prevent memory issues...")
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    # Feature selection
    features = ["Voltage_measured", "Current_measured", "Power"]
    target = "Temperature_measured"

    X = df[features].fillna(0)
    y = df[target].fillna(0)

    print("📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define algorithms with safer settings
    algorithms = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=50,       # reduced from 100
            max_depth=10,          # limit depth to save memory
            max_samples=0.3,       # train each tree on 30% subsample
            n_jobs=-1,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            n_jobs=-1,
            random_state=42,
            tree_method="hist"  # faster and less memory-hungry
        ),
        # "SVM": SVR(kernel="linear")  # linear kernel uses much less memory but takes more time 
        
        
        
        
        
        
        # "RandomForest": RandomForestRegressor(
        #     n_estimators=300,
        #     max_depth=None,       # allow full growth
        #     max_features='sqrt',  # random feature selection per split
        #     min_samples_leaf=2,
        #     n_jobs=-1,
        #     random_state=42
        # ),
        # "LightGBM": LGBMRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.03,
        #     num_leaves=64,
        #     feature_fraction=0.8,
        #     bagging_fraction=0.8,
        #     bagging_freq=5,
        #     n_jobs=-1,
        #     random_state=42
        # ),
        # "XGBoost": XGBRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.03,
        #     max_depth=10,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     tree_method="hist",
        #     n_jobs=-1,
        #     random_state=42
        # )

        
        
        
        
        
        
        
        
    }

    metrics = {}
    best_rmse = float("inf")
    best_model_name = None
    best_model = None

    print("🚀 Training all models...")
    for name, model in algorithms.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            print(f"{name} RMSE: {rmse:.4f}")
            metrics[name] = rmse

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_model = model
        except MemoryError:
            print(f"❌ Skipping {name} due to memory error.")
        except Exception as e:
            print(f"⚠️ Error training {name}: {e}")

    if best_model is None:
        print("❌ No model trained successfully.")
        return

    print(f"\n✅ Best model: {best_model_name} with RMSE={best_rmse:.4f}")

    # Save best model
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"💾 Best model saved at {MODEL_DIR}best_model.pkl")

    # Save metrics
    metrics["best_model"] = best_model_name
    metrics["best_rmse"] = best_rmse
    metrics["n_train_samples"] = len(X_train)
    metrics["n_test_samples"] = len(X_test)
    metrics["n_features"] = len(features)
    metrics["train_time"] = str(datetime.now())

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📄 Metrics saved at {METRICS_PATH}")


if __name__ == "__main__":
    train_all_models()
