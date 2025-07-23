import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

# Load dataset
df = pd.read_csv("TrainingDatasetRainFallPrediction.csv")

# Convert categorical columns (Location) to numerical
df["Location"] = df["Location"].astype("category").cat.codes

# Feature selection
features = ["Temperature (C)", "Humidity (%)", "Wind Speed (km/h)", "Pressure (hPa)", "Cloud Cover (%)"]
X = df[features]

# Target variables
y_rain = (df["Rainfall (mm)"] > 0).astype(int)  # Binary classification
y_rainfall_intensity = df["Rainfall Intensity (mm/hr)"]  # Regression
y_flood = (df["Flood Occurrence Rate"] > 0.1).astype(int)  # Binary classification
y_agriculture = (df["Agriculture Suitability Rate"] > 0.7).astype(int)  # Binary classification

# Split data for training
X_train, X_test, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)
X_train_rf, X_test_rf, y_train_flood, y_test_flood = train_test_split(X, y_flood, test_size=0.2, random_state=42)
X_train_ar, X_test_ar, y_train_agri, y_test_agri = train_test_split(X, y_agriculture, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_rainfall, y_test_rainfall = train_test_split(X, y_rainfall_intensity, test_size=0.2, random_state=42)

# Train models
rain_model = LogisticRegression()
rain_model.fit(X_train, y_train_rain)

flood_model = RandomForestClassifier(n_estimators=100)
flood_model.fit(X_train_rf, y_train_flood)

agriculture_model = RandomForestClassifier(n_estimators=100)
agriculture_model.fit(X_train_ar, y_train_agri)

rainfall_model = RandomForestRegressor(n_estimators=100)
rainfall_model.fit(X_train_r, y_train_rainfall)

rain_accuracy = accuracy_score(y_test_rain, rain_model.predict(X_test))
flood_accuracy = accuracy_score(y_test_flood, flood_model.predict(X_test_rf))
agri_accuracy = accuracy_score(y_test_agri, agriculture_model.predict(X_test_ar))
rainfall_mae = mean_absolute_error(y_test_rainfall, rainfall_model.predict(X_test_r))

# Streamlit App
st.title("Rainfall Prediction Model")
st.title("Training Phase")
st.subheader("Model Performance on the dataset")
st.subheader("Below are the Model Evaluation Metrics: Accuracy Scores")
st.metric(label="🌧️ Rain Prediction Accuracy", value=f"{rain_accuracy:.2%}")
st.metric(label="🌊 Flood Prediction Accuracy", value=f"{flood_accuracy:.2%}")
st.metric(label="🌾 Agriculture Suitability Accuracy", value=f"{agri_accuracy:.2%}")

st.subheader("Rainfall Intensity Error")
st.metric(label="📏 Rainfall Intensity MAE", value=f"{rainfall_mae:.2f}")

# Optional: Use columns for better UI
col1, col2 = st.columns(2)
col1.metric("Rain Prediction Accuracy", f"{rain_accuracy:.2%}")
col2.metric("Flood Prediction Accuracy", f"{flood_accuracy:.2%}")

col1.metric("Agriculture Suitability Accuracy", f"{agri_accuracy:.2%}")
col2.metric("Rainfall Intensity MAE", f"{rainfall_mae:.2f}")

# Streamlit App

# Data Visualizations
st.title("Visualizations of Model analysis on trainig data")

accuracies = [accuracy_score(y_test_rain, rain_model.predict(X_test)),
              accuracy_score(y_test_flood, flood_model.predict(X_test_rf)),
              accuracy_score(y_test_agri, agriculture_model.predict(X_test_ar))]

labels = ["Rain", "Flood", "Agriculture Suitability"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=labels, y=accuracies, palette="coolwarm", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# Regression graph: Actual vs Predicted Rainfall Intensity
st.subheader("Rainfall Intensity: Actual vs Predicted")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test_rainfall, rainfall_model.predict(X_test_r), color="blue", alpha=0.6)
ax.plot([min(y_test_rainfall), max(y_test_rainfall)], [min(y_test_rainfall), max(y_test_rainfall)], color="red", linestyle="--")
ax.set_xlabel("Actual Rainfall Intensity (mm/hr)")
ax.set_ylabel("Predicted Rainfall Intensity (mm/hr)")
st.pyplot(fig)

# Classification Reports

st.subheader("Classification Reports")

# st.text("Rain Prediction Report:\n" + classification_report(y_test_rain, rain_model.predict(X_test)))
# st.text("Flood Prediction Report:\n" + classification_report(y_test_flood, flood_model.predict(X_test_rf)))
# st.text("Agriculture Suitability Report:\n" + classification_report(y_test_agri, agriculture_model.predict(X_test_ar)))


def classification_report_to_df(report):
    report_dict = classification_report(report['y_true'], report['y_pred'], output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(2)  # Round values for better readability
    return df

# Generate reports for each model
rain_report = classification_report_to_df({'y_true': y_test_rain, 'y_pred': rain_model.predict(X_test)})
flood_report = classification_report_to_df({'y_true': y_test_flood, 'y_pred': flood_model.predict(X_test_rf)})
agri_report = classification_report_to_df({'y_true': y_test_agri, 'y_pred': agriculture_model.predict(X_test_ar)})

# Streamlit UI
st.title("Weather & Agriculture Prediction Model")

st.subheader("Classification Reports")

# Create tabs for each classification report
tab1, tab2, tab3 = st.tabs(["🌧️ Rain Prediction", "🌊 Flood Prediction", "🌾 Agriculture Suitability"])

with tab1:
    st.write("### Rain Prediction Report")
    st.dataframe(rain_report.style.format("{:.2f}"))  # Neatly formatted DataFrame

with tab2:
    st.write("### Flood Prediction Report")
    st.dataframe(flood_report.style.format("{:.2f}"))

with tab3:
    st.write("### Agriculture Suitability Report")
    st.dataframe(agri_report.style.format("{:.2f}"))

# Prediction based on input
# Sidebar for user input
st.sidebar.header("Enter Weather Details")
temp = st.sidebar.number_input("Temperature (C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
pressure = st.sidebar.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1010.0)
cloud_cover = st.sidebar.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)




# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[temp, humidity, wind_speed, pressure, cloud_cover]])

    # Make predictions
    rain_pred = rain_model.predict(input_data)[0]
    rainfall_intensity_pred = rainfall_model.predict(input_data)[0]
    flood_pred = flood_model.predict(input_data)[0]
    agriculture_pred = agriculture_model.predict(input_data)[0]

    # Get probability function
    def get_probability(model, input_data):
        prob = model.predict_proba(input_data)[0]
        return prob[1] if len(prob) > 1 else prob[0]

    rain_prob = get_probability(rain_model, input_data)
    flood_prob = get_probability(flood_model, input_data)
    agriculture_prob = get_probability(agriculture_model, input_data)

    # Display Predictions
    st.title("Prediction Results")
    st.write(f"**Rain Expected:** {'Yes' if rain_pred == 1 else 'No'}")
    st.write(f"**Predicted Rainfall Intensity:** {rainfall_intensity_pred:.2f} mm/hr")
    st.write(f"**Flood Occurrence:** {'Likely' if flood_pred == 1 else 'Unlikely'}")
    st.write(f"**Agriculture Suitability:** {'Suitable' if agriculture_pred == 1 else 'Not Suitable'}")

    # Prediction Probability Bar Chart
    st.subheader("Weather Prediction Probability")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=[rain_prob, flood_prob, agriculture_prob], palette="coolwarm", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Save to CSV
    new_data = pd.DataFrame([{
        "Temperature (C)": temp,
        "Humidity (%)": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Pressure (hPa)": pressure,
        "Cloud Cover (%)": cloud_cover,
        "Rain Expected": "Yes" if rain_pred == 1 else "No",
        "Predicted Rainfall Intensity (mm/hr)": round(rainfall_intensity_pred, 2),
        "Flood Occurrence": "Likely" if flood_pred == 1 else "Unlikely",
        "Agriculture Suitability": "Suitable" if agriculture_pred == 1 else "Not Suitable"
    }])

    new_data.to_csv("Output.csv", index=False)
    st.success("Prediction results saved to Output.csv")
