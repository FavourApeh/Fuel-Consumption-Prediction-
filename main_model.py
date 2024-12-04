# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
dataset = pd.read_csv('Fuel_Consumption.csv')

# Check for missing values
missing_values = dataset.isnull().sum()
print("Missing values in the dataset:\n", missing_values)

# Remove duplicates
dataset_no_duplicates = dataset.drop_duplicates()

# Save the cleaned dataset
dataset_no_duplicates.to_csv('cleaned_dataset.csv', index=False)

# Load the cleaned dataset
cleaned_dataset = pd.read_csv('cleaned_dataset.csv')

# Drop irrelevant and redundant columns
columns_to_drop = ['Make', 'Model', 'Fuel Consumption City (L/100 km)',
                   'Fuel Consumption Hwy (L/100 km)']
cleaned_dataset = cleaned_dataset.drop(columns=columns_to_drop)

# Define numeric and categorical columns
numeric_columns = ['Engine Size(L)', 'Cylinders', 'CO2 Emissions(g/km)']
categorical_columns = ['Vehicle Class', 'Fuel Type', 'Transmission']

# Separate features and target
target = 'Fuel Consumption Comb (L/100 km)'
X = cleaned_dataset.drop(columns=[target])
y = cleaned_dataset[target]

# OneHotEncode categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_array = encoder.fit_transform(X[categorical_columns])

# Save the encoder
joblib.dump(encoder, "encoder.pkl")

# Convert the encoded array to a DataFrame
encoded_columns = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=X.index)

# Combine numeric and encoded categorical data
X = pd.concat([X[numeric_columns], encoded_df], axis=1)

# Scale numeric features
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate models
print("\nModel Performance:")
for model, name in [(lr_model, "Linear Regression"), (rf_model, "Random Forest")]:
    y_pred = model.predict(X_test)
    print(f"{name}:")
    print(f"  Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred): .2f}")
    print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred): .2f}")
    print(f"  R-squared (R²): {r2_score(y_test, y_pred): .2f}")

joblib.dump(X_train.columns.tolist(), "trained_features.pkl")

# Save both models
joblib.dump(lr_model, "linear_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")

print("\nModels and preprocessing objects have been saved successfully!")
