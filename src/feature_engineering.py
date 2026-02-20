# src/feature_engineering.py

import pandas as pd
from src.load_data import load_all_csv

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the battery dataset and creates new features.
    Steps:
    1. Remove unnecessary columns
    2. Drop duplicates
    3. Handle missing values
    4. Feature engineering (Power, optional rolling features)
    """
    df = df.copy()

    # 1️⃣ Remove unnecessary columns
    columns_to_drop = ["Current_charge", "Voltage_charge", "Unnamed: 0"]  # adjust if more exist
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 2️⃣ Drop duplicates
    df = df.drop_duplicates()

    # 3️⃣ Handle missing values
    # Fill missing numeric values with 0
    df = df.fillna(0)

    # 4️⃣ Feature engineering
    if "Voltage_measured" in df.columns and "Current_measured" in df.columns:
        df["Power"] = df["Voltage_measured"] * df["Current_measured"]

    # 5️⃣ Optional: rolling averages for smoothing (window=10)
    if "Voltage_measured" in df.columns:
        df["Voltage_roll_mean"] = df["Voltage_measured"].rolling(window=10).mean().fillna(0)
    if "Current_measured" in df.columns:
        df["Current_roll_mean"] = df["Current_measured"].rolling(window=10).mean().fillna(0)

    return df


if __name__ == "__main__":
    print("📂 Loading all CSV files...")
    df = load_all_csv()
    print(f"✅ Original data shape: {df.shape}")

    df = clean_and_engineer_features(df)
    print(f"✨ Data after cleaning and feature engineering: {df.shape}")

    print("🔍 Columns available:", df.columns.tolist()[:10], "...")
    print("✅ Null values per column:\n", df.isnull().sum())
