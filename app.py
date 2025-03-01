import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the SVM regression model
svm_regressor = SVR(kernel='rbf')
svm_regressor.fit(X_scaled, y)

# Streamlit UI for user input
st.title("California Housing Price Prediction with SVM Regression")
st.write("This app predicts the housing price based on several features.")

# Input fields for user to enter feature values
avg_rooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=10.0, step=0.1)
avg_income = st.number_input("Average Income per Household ($10,000)", min_value=1.0, max_value=15.0, step=0.1)
avg_house_age = st.number_input("Average House Age (years)", min_value=1, max_value=100, step=1)
population = st.number_input("Population", min_value=1, max_value=5000, step=1)
households = st.number_input("Number of Households", min_value=1, max_value=3000, step=1)
latitude = st.number_input("Latitude", min_value=32.0, max_value=34.5, step=0.1)
longitude = st.number_input("Longitude", min_value=-118.5, max_value=-116.5, step=0.1)

# Prepare the input data as a numpy array
user_input = np.array([[avg_rooms, avg_income, avg_house_age, population, households, latitude, longitude]])
user_input_scaled = scaler.transform(user_input)

# Make the prediction
prediction = svm_regressor.predict(user_input_scaled)

# Display the predicted housing price
st.write(f"The predicted housing price is: **${prediction[0]:,.2f}**")

# Option to show the model performance metrics (e.g., R², MSE)
if st.checkbox("Show Model Performance Metrics"):
    from sklearn.metrics import mean_squared_error, r2_score

    # Predictions on the test set (using the entire dataset for demonstration)
    y_pred = svm_regressor.predict(X_scaled)

    # Model performance metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")

    # Plot true vs predicted values
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Housing Prices (SVM Regression)')
    st.pyplot(plt)
