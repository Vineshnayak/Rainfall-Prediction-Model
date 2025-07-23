import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Load training dataset
train_df = pd.read_csv("TrainingDatasetRainFallPrediction.csv")

# Convert categorical columns (Location) if present
if "Location" in train_df.columns:
    train_df["Location"] = train_df["Location"].astype("category").cat.codes

# Feature selection
features = ["Temperature (C)", "Humidity (%)", "Wind Speed (km/h)", "Pressure (hPa)", "Cloud Cover (%)"]
X_train = train_df[features]

# Target variables
y_rain = (train_df["Rainfall (mm)"] > 0).astype(int)  # Binary classification
y_rainfall_intensity = train_df["Rainfall Intensity (mm/hr)"]  # Regression
y_flood = (train_df["Flood Occurrence Rate"] > 0.1).astype(int)  # Binary classification
y_agriculture = (train_df["Agriculture Suitability Rate"] > 0.7).astype(int)  # Binary classification

# Train models
rain_model = LogisticRegression()
rain_model.fit(X_train, y_rain)

flood_model = RandomForestClassifier(n_estimators=100)
flood_model.fit(X_train, y_flood)

agriculture_model = RandomForestClassifier(n_estimators=100)
agriculture_model.fit(X_train, y_agriculture)

rainfall_model = RandomForestRegressor(n_estimators=100)
rainfall_model.fit(X_train, y_rainfall_intensity)

# Streamlit UI
st.title("Rainfall Intensity, Agriculture suitability and Flood Prediction")
st.write("Click the button below to predict on the sample test dataset.")

if st.button("Predict on the set of sample test cases"):
    # Load testing dataset
    test_df = pd.read_csv("weather_testing_data.csv")

    # Prepare input features for prediction
    X_test = test_df[features]

    # Make predictions
    test_df["Rainfall (mm)"] = rain_model.predict(X_test)
    test_df["Rainfall Intensity (mm/hr)"] = rainfall_model.predict(X_test)
    test_df["Flood Occurrence Rate"] = flood_model.predict(X_test)
    test_df["Agriculture Suitability Rate"] = agriculture_model.predict(X_test)

    # Save the final results
    test_df.to_csv("predictions_output.csv", index=False)
    st.success("Final predictions saved in 'predictions_output.csv' successfully!")

    # Display the dataset
    st.write("### Prediction Results:")
    st.dataframe(test_df)

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Rain", "Flood", "Agriculture Suitability"], 
                y=[test_df["Rainfall (mm)"].mean(), 
                   test_df["Flood Occurrence Rate"].mean(), 
                   test_df["Agriculture Suitability Rate"].mean()], 
                palette="coolwarm")
    plt.title("Average Prediction Results")
    plt.ylabel("Probability")
    st.pyplot(plt)
