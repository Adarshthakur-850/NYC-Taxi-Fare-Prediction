import pandas as pd
import os

def evaluate_models_and_save(results_df):
    print("Saving evaluation results...")
    if not os.path.exists("models"):
        os.makedirs("models")
        
    results_df.to_csv("models/metrics.csv", index=False)
    print(results_df)
