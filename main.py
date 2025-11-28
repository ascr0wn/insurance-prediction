import pandas as pd
import os

# Define path
file_path = os.path.join("data", "raw", "insurance.csv")

try:
    df = pd.read_csv(file_path)
    print("✅ SUCCESS: Data loaded!")
    print(f"Shape: {df.shape} (Rows, Columns)")
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ ERROR: File not found. Check your 'data/raw' folder.")
