import pandas as pd
import os
import requests

DATA_URL = "https://raw.githubusercontent.com/krishnaik06/NYC-Taxi-Fares-Prediction/master/taxifare.csv"
DATA_PATH = os.path.join("data", "taxi_fare.csv")

def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading sample dataset...")
        try:
            response = requests.get(DATA_URL, stream=True)
            response.raise_for_status()
            
            with open(DATA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
            
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, nrows=100000)
    print(f"Dataset shape: {df.shape}")
    return df
