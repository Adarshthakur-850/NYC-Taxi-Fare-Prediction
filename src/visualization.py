import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_clusters(df):
    print("Visualizing clusters (Maps)...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='pickup_longitude', y='pickup_latitude', 
        hue='pickup_cluster', palette='tab10', 
        data=df.sample(min(10000, len(df))), 
        alpha=0.6, s=10
    )
    plt.title("Pickup Clusters Map")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/pickup_clusters_map.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='dropoff_longitude', y='dropoff_latitude', 
        hue='dropoff_cluster', palette='tab10', 
        data=df.sample(min(10000, len(df))), 
        alpha=0.6, s=10
    )
    plt.title("Dropoff Clusters Map")
    plt.savefig("plots/dropoff_clusters_map.png", bbox_inches='tight')
    plt.close()

def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=results_df)
    plt.title("Model Comparison (RMSE)")
    plt.ylabel("Root Mean Squared Error")
    plt.savefig("plots/model_comparison_rmse.png")
    plt.close()
