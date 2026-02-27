from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import pandas as pd
import numpy as np

def train_models(df):
    print("Training models...")
    
    target = 'fare_amount'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    results = []
    best_rmse = float('inf')
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name
            
    print(f"Best Model: {best_name} (RMSE: {best_rmse:.2f})")
    if best_model:
        joblib.dump(best_model, "models/best_model.pkl")
        
    return pd.DataFrame(results), best_model
