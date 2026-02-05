
import os
import pandas as pd
import time

upload_folder = 'uploads'
files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
print(f"Files found: {files}")

if not files:
    print("No files!")
    exit()

# Logic from app.py (updated to getmtime)
latest_file_name = max([os.path.join(upload_folder, f) for f in files], key=os.path.getmtime)
print(f"Selected file: {latest_file_name}")
print(f"Size: {os.path.getsize(latest_file_name)} bytes")

try:
    df = pd.read_csv(latest_file_name)
    print(f"DataFrame shape: {df.shape}")
    print(df.head())
    
    count = 0
    for index, row in df.iterrows():
        count += 1
    print(f"Iterated {count} rows")
    
except Exception as e:
    print(f"Error reading CSV: {e}")
