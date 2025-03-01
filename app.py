import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fetch COVID-19 data for USA from the API
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Plot COVID-19 Data (Total Cases, Active, Recovered, Deaths)
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generate random historical data for the last 30 days (for demonstration purposes)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Use SVR (Support Vector Regression) for predicting the cases
X = df_historical[["day"]]
y = df_historical["cases"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVR model
svr_model = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Predict cases for the next day (Day 31)
next_day = np.array([[31]])
predicted_cases = svr_model.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Model evaluation: Mean Squared Error and R-squared Score
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plot true vs predicted cases
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Cases')
plt.plot(X_test, y_pred, color='red', label='Predicted Cases', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Cases')
plt.title('True vs Predicted COVID-19 Cases (SVM Regression)')
plt.legend()
plt.show()

# Streamlit App Interface
st.title("COVID-19 Cases Prediction-in USA with SVM Regression")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User input for day number
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = svr_model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")

    # Show model evaluation metrics if the user clicks the button
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")
    
    # Display a plot of true vs predicted cases
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Cases')
    ax.plot(X_test, y_pred, color='red', label='Predicted Cases', linewidth=2)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cases')
    ax.set_title('True vs Predicted COVID-19 Cases (SVM Regression)')
    ax.legend()
    st.pyplot(fig)
