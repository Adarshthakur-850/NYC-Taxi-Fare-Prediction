import pandas as pd
import numpy as np

def engineer_features(df):
    print("Engineering features...")
    
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['weekday'] = df['pickup_datetime'].dt.dayofweek
    df['year'] = df['pickup_datetime'].dt.year
    
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    
    final_cols = [
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 
        'distance_km',
        'pickup_cluster', 'dropoff_cluster',
        'hour', 'day', 'month', 'weekday', 'year', 'is_rush_hour',
        'fare_amount'
    ]
    
    return df[final_cols]
