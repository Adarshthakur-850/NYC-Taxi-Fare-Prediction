import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df):
    print("Performing EDA...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.figure(figsize=(10, 6))
    sns.histplot(df['fare_amount'], bins=100, kde=True, color='green')
    plt.title("Fare Amount Distribution")
    plt.xlim(0, 100)
    plt.savefig("plots/fare_distribution.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='distance_km', y='fare_amount', data=df.sample(min(10000, len(df))), alpha=0.3)
    plt.title("Distance vs Fare Amount")
    plt.xlim(0, 50)
    plt.ylim(0, 150)
    plt.savefig("plots/distance_vs_fare.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    hourly_fare = df.groupby('hour')['fare_amount'].mean()
    sns.lineplot(x=hourly_fare.index, y=hourly_fare.values, marker='o')
    plt.title("Average Fare by Hour of Day")
    plt.grid(True)
    plt.savefig("plots/fare_by_hour.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.hist2d(df['pickup_longitude'], df['pickup_latitude'], bins=100, cmap='hot', cmin=1)
    plt.colorbar(label='Count')
    plt.title("Pickup Location Heatmap")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig("plots/pickup_heatmap.png")
    plt.close()
