# Fuel-Consumption-Prediction-
# Overview

This project predicts vehicle fuel consumption (in L/100 km) using machine learning models. The prediction is based on various vehicle characteristics, including engine size, cylinder count, CO2 emissions, and other specifications. The application is deployed via a Flask API for real-time predictions.

# Features

Machine Learning Models: Linear Regression and Random Forest Regressor.
End-to-End Workflow: Data cleaning, feature engineering, model training, evaluation, and deployment.
API Deployment: Flask-based API for easy integration with other applications.

# Dataset Details

The dataset includes the following key features:
**Fuel Consumption**
City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km).
The combined rating (55% city, 45% highway) is available in both L/100 km and miles per imperial gallon (mpg).
**CO2 Emissions**
The tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving.
**Vehicle Specifications**
Features like engine size, number of cylinders, transmission type, fuel type, and vehicle class.

# Project Pipeline

**1. Data Preprocessing**
Removed duplicate entries and irrelevant columns.
One-hot encoded categorical variables (e.g., Vehicle Class, Fuel Type, Transmission).
Scaled numeric features using StandardScaler.

**2. Model Training**
Trained a Linear Regression and a Random Forest Regressor.
Evaluated models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

**3. Model Deployment**
Flask-based API with endpoints for predictions:
/predict_lr: Predict fuel consumption using Linear Regression.
/predict_rf: Predict fuel consumption using Random Forest.
