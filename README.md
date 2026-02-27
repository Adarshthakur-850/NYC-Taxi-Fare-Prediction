# 🚖 NYC Taxi Fare Prediction

A Machine Learning project that predicts taxi fares in New York City using historical trip data. The model estimates fare amount based on trip distance, pickup/dropoff coordinates, passenger count, and temporal features.

---

## 📌 Problem Statement

Taxi fare estimation is essential for:

* Transparent pricing
* Fraud detection
* Dynamic pricing optimization
* Ride cost prediction before booking

The objective of this project is to build a regression model that accurately predicts the taxi fare amount using structured trip data.

---

## 📊 Dataset

The dataset consists of historical NYC taxi trips including:

* Pickup datetime
* Pickup longitude & latitude
* Dropoff longitude & latitude
* Passenger count
* Fare amount (target variable)

Data preprocessing included:

* Removing invalid coordinates
* Filtering out extreme outliers
* Handling missing values
* Feature engineering

---

## ⚙️ Feature Engineering

Key engineered features:

* Trip distance (Haversine formula)
* Pickup hour
* Day of week
* Weekend indicator
* Peak hour indicator

Distance calculation significantly improved model performance.

---

## 🧠 Model Architecture

This is a supervised regression problem.

Models evaluated:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

Final model selected based on:

* RMSE
* MAE
* Cross-validation performance

---

## 📈 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Model performance demonstrates strong generalization on unseen data.

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 📂 Project Structure

```
NYC-Taxi-Fare-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── EDA_and_Model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Adarshthakur-850/NYC-Taxi-Fare-Prediction.git
cd NYC-Taxi-Fare-Prediction
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Model

```bash
python src/train.py
```

---

## 🔍 Making Predictions

```bash
python src/predict.py
```

---

## 📊 Example Prediction Flow

Input:

* Pickup: Manhattan
* Dropoff: Brooklyn
* Passengers: 2
* Time: 6 PM

Output:

* Estimated Fare: $XX.XX

---

## 🧪 Future Improvements

* Hyperparameter tuning using Optuna
* Model deployment using FastAPI
* Docker containerization
* CI/CD integration
* Real-time prediction API
* Kubernetes deployment
* Monitoring with Prometheus + Grafana

---

## 📌 Key Learnings

* Geospatial feature engineering
* Regression model optimization
* Handling large datasets
* Bias-variance tradeoff
* Feature importance analysis

---

## 📄 License

This project is open-source and available under the MIT License.
