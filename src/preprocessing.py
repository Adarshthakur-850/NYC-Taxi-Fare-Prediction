import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def preprocess_data(df):
    print("Preprocessing data...")
    
    df = df.dropna()
    
    if 'pickup_datetime' in df.columns:
        if df['pickup_datetime'].dtype == 'object':
            df['pickup_datetime'] = df['pickup_datetime'].astype(str).str.replace(' UTC', '')
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            
    df = df[
        (df['fare_amount'] > 0) & 
        (df['fare_amount'] < 500) &
        (df['pickup_longitude'].between(-75, -72)) &
        (df['pickup_latitude'].between(40, 42)) &
        (df['dropoff_longitude'].between(-75, -72)) &
        (df['dropoff_latitude'].between(40, 42)) &
        (df['passenger_count'] > 0) &
        (df['passenger_count'] < 8)
    ]
    
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    df = df[df['distance_km'] > 0.05]
    
    print(f"Shape after cleaning: {df.shape}")
    return df