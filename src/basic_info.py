# src/basic_info.py
from load_data import load_all_csv

# Load the combined dataframe
df = load_all_csv()

# Basic info
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
