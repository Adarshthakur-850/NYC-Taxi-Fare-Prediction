# NYC Taxi Fare Prediction using Geolocation Clustering

## Overview
A complete end-to-end project to predict taxi fare amounts using trip details and geolocation clustering (K-Means) to capture spatial patterns.

## Structure
- `src/`: Core logic modules
- `data/`: Dataset storage
- `models/`: Saved models
- `plots/`: Generated visualizations

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py`

## Features
- **Geolocation Clustering**: K-Means clustering on pickup and dropoff coordinates.
- **Feature Engineering**: Time-based features, Haversine distance.
- **Models**: Linear Regression, Random Forest, XGBoost.
