# src/load_data.py
import os
import pandas as pd

DATA_FOLDER = r"C:\Users\vasan\OneDrive\Desktop\battery_health_project\data"

def load_all_csv(data_folder=DATA_FOLDER):
    all_dfs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_folder, filename)
            try:
                df = pd.read_csv(filepath)
                df["source_file"] = filename
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    return pd.concat(all_dfs, ignore_index=True)

# This block runs only if you execute this file directly
if __name__ == "__main__":
    df = load_all_csv()
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

