import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate some sample data
# Assume we have 1000 samples with features: RPM, Torque, Power, Wheel Radius, Incline Angle, Friction Coefficient
np.random.seed(42)
rpm = np.random.uniform(100, 5000, 1000)  # RPM values
torque = np.random.uniform(10, 100, 1000)  # Torque values
power = np.random.uniform(100, 1000, 1000)  # Power output values
wheel_radius = np.random.uniform(0.2, 1.0, 1000)  # Wheel radius in meters
incline_angle = np.random.uniform(0, 15, 1000)  # Incline angle in degrees
friction_coefficient = np.random.uniform(0.01, 0.3, 1000)  # Friction coefficient

# Assume a synthetic function for weight based on the input parameters
weight = (0.2 * rpm + 0.5 * torque + 0.1 * power + 15 * wheel_radius - 2 * incline_angle + 100 * friction_coefficient) + np.random.normal(0, 20, 1000)

# Stack features together for input to the model
X = np.column_stack((rpm, torque, power, wheel_radius, incline_angle, friction_coefficient))
y = weight  # The target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Neural Network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),  # 64 neurons in first hidden layer
    Dense(32, activation='relu'),  # 32 neurons in second hidden layer
    Dense(16, activation='relu'),  # 16 neurons in third hidden layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test MAE: {test_mae}")

# Make predictions
predictions = model.predict(X_test)

# Sample output for comparison
for i in range(5):
    print(f"Predicted weight: {predictions[i][0]:.2f}, Actual weight: {y_test[i]:.2f}")
