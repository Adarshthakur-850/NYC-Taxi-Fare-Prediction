from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def perform_clustering(df, n_clusters=10):
    print(f"Performing Clustering (K={n_clusters})...")
    
    pickup_coords = df[['pickup_latitude', 'pickup_longitude']]
    kmeans_pickup = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['pickup_cluster'] = kmeans_pickup.fit_predict(pickup_coords)
    
    dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']]
    kmeans_dropoff = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['dropoff_cluster'] = kmeans_dropoff.fit_predict(dropoff_coords)
    
    return df, kmeans_pickup, kmeans_dropoff
