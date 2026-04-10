# Rainfall Prediction Model

## Overview

The **Rainfall Prediction Model** is a machine learning web application built with Streamlit. It uses historical weather data and live API feeds to provide predictions for weather patterns, flood risks, and agricultural suitability. Using classification and regression algorithms, this tool can assist meteorologists, agricultural planners, and disaster management teams.

## Features

- **Multi-Objective Predictions**: 
  - **Rain Prediction (Binary)**: Determines the probability of rain occurrence using Logistic Regression.
  - **Rainfall Intensity (Regression)**: Estimates the expected rainfall in mm/hr using Random Forest Regressor.
  - **Flood Risk Classification (Binary)**: Flags high-risk flood conditions using Random Forest Classifier.
  - **Agriculture Suitability (Binary)**: Recommends optimal planting conditions.
- **Dynamic Input Mechanisms**:
  - **Manual Entry**: Users can adjust specific weather features (Temperature, Humidity, Wind Speed, Pressure, Cloud Cover) using interactive UI sliders.
  - **Live Fetching**: Integrates with the **OpenWeather API** to fetch real-time weather data for a given city.
- **Explainable AI (XAI)**: Includes **SHAP (SHapley Additive exPlanations)** to break down and visualize feature importance, specifically clarifying the model's rationale behind flood predictions.
- **Alerts & Recommendations**:
  - **Alert System**: Provides alerts (critical, moderate, low) based on flood occurrence probabilities.
  - **Crop Recommendations**: Suggests crops (e.g., Rice, Sugarcane, Wheat, Cotton) based on current temperature and humidity thresholds if conditions are suitable.
- **Data Export**: Saves prediction results into an `Output.csv` file.

## Tech Stack & Dependencies

The project predominantly uses Python for data processing, machine learning, and frontend interaction.

- **Frontend App**: `streamlit`
- **Data Handling**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Data Visualization**: `matplotlib`, `seaborn`, `shap`
- **API Requests**: `requests`

*See `requirements.txt` for the full dependency specifications.*

## Installation & Setup

1. **Clone the repository or navigate to the project directory:**
   ```bash
   cd /path/to/RainfallPrediction
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install the required packages:**
   ```bash
   # Make sure streamlit and shap are installed if not already in requirements.txt
   pip install -r requirements.txt
   pip install streamlit requests shap
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run RainfallPrediction.py
   ```

## Usage

1. **Dashboard Overview:** Upon launching, the application will display the model training metrics (Accuracy, Mean Absolute Error) and performance graphs across different algorithms.
2. **Making Predictions:** On the sidebar, choose between:
   - **Manual Entry**: Adjust the sliders for an isolated simulation.
   - **Live OpenWeather API**: Provide your OpenWeather API Key and a Target City to pull real-time characteristics.
3. **Analyze Results:** After hitting **Predict**, view the detailed readouts, the AI's explanation behind the predictions, crop recommendations, and safety alerts. 

## Project Structure

- `RainfallPrediction.py`: The core application file housing the data pipeline, model training, prediction logic, and Streamlit user interface.
- `TrainingDatasetRainFallPrediction.csv`: The primary dataset utilized for training the models.
- `Output.csv` / `predictions_output.csv`: Stored outputs of previous prediction queries.
- `requirements.txt`: Python package dependencies.

## Acknowledgements 

- Scikit-learn for robust ML implementations.
- OpenWeather API for real-time live data access.
- SHAP for providing model explainability.
