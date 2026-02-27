import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.clustering import perform_clustering
from src.feature_engineering import engineer_features
from src.eda import perform_eda
from src.model_trainer import train_models
from src.evaluation import evaluate_models_and_save
from src.visualization import visualize_clusters, plot_model_comparison

def main():
    print("Starting NYC Taxi Fare Prediction Pipeline...")
    
    try:
        df = load_data()
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    df = preprocess_data(df)
    
    df, _, _ = perform_clustering(df)
    
    df = engineer_features(df)
    
    try:
        perform_eda(df)
    except Exception as e:
        print(f"EDA error (skipping): {e}")

    results_df, best_model = train_models(df)
    
    evaluate_models_and_save(results_df)
    
    try:
        visualize_clusters(df)
        plot_model_comparison(results_df)
    except Exception as e:
        print(f"Visualization error: {e}")
        
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline Failed: {e}")
