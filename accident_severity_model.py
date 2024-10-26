import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('road_accident_data.csv')

# Define independent variables (features) and dependent variable (target)
X = df[['Weather', 'Road_Type', 'Traffic_Volume', 'Time_of_Day', 'Driver_Behavior']]
y = df['Accident_Severity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the trained model for future use
joblib.dump(model, 'road_accident_severity_model.pkl')
print("Model saved as 'road_accident_severity_model.pkl'")

# Example prediction using a new data point
new_data = [[1, 2, 250, 10, 3]]  # Hypothetical data point
severity_prediction = model.predict(new_data)
print("Predicted Accident Severity for new data:", severity_prediction[0])
