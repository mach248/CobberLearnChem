import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the DataFrame with all compounds
all_data = {
    'Compound': ['Methane', 'Water', 'Propane', 'Ethanol', 'Formic Acid',
                 'Acetic Acid', 'Butane', 'Acetone', 'Benzene', 'Toluene', 'Octane'],
    'MW': [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    'BoilingPoint': [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}

# Create a DataFrame without Butane for training
data = {
    'Compound': ['Methane', 'Water', 'Propane', 'Ethanol', 'Formic Acid',
                 'Acetic Acid', 'Acetone', 'Benzene', 'Toluene', 'Octane'],
    'MW': [16, 18, 44, 46, 46, 60, 58, 78, 92, 114],
    'BoilingPoint': [-161, 100, -42, 78, 101, 118, 56, 80, 111, 125]
}

# Create the training DataFrame (without Butane)
df = pd.DataFrame(data)
print("Training data (without Butane):")
print(df)

# Define X (input) and y (target) for training
X = df[['MW']].values  # Note: X needs to be a 2D array for sklearn
y = df['BoilingPoint'].values

# Extract Butane data for testing
butane_data = pd.DataFrame({
    'Compound': ['Butane'],
    'MW': [58],
    'BoilingPoint': [-1]
})
print("\nTest data (Butane only):")
print(butane_data)

X_butane = butane_data[['MW']].values
y_butane_actual = butane_data['BoilingPoint'].values[0]

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Train a neural network model
nn_model = MLPRegressor(
    hidden_layer_sizes=(15, 7),  # Two hidden layers with 10 neurons each
    activation='relu',            # ReLU activation function
    max_iter=5000,                # Maximum number of iterations
    random_state=42,              # For reproducibility
    verbose=True                  # To see training progress
)
nn_model.fit(X, y)

# Make predictions on the training data
y_pred_lr = lr_model.predict(X)
y_pred_nn = nn_model.predict(X)

# Calculate metrics for training data
mae_lr = mean_absolute_error(y, y_pred_lr)
mse_lr = mean_squared_error(y, y_pred_lr)
r2_lr = r2_score(y, y_pred_lr)

mae_nn = mean_absolute_error(y, y_pred_nn)
mse_nn = mean_squared_error(y, y_pred_nn)
r2_nn = r2_score(y, y_pred_nn)

print(f"\nTraining Results:")
print(f"Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
print(f"Neural Network - MAE: {mae_nn:.2f}, MSE: {mse_nn:.2f}, R²: {r2_nn:.2f}")
print(f"Neural Network iterations used: {nn_model.n_iter_}")

# Predict Butane's boiling point using both models
butane_pred_lr = lr_model.predict(X_butane)[0]
butane_pred_nn = nn_model.predict(X_butane)[0]

print(f"\nButane Prediction Results:")
print(f"Actual Boiling Point: {y_butane_actual}°C")
print(f"Linear Regression Prediction: {butane_pred_lr:.2f}°C (Error: {abs(butane_pred_lr - y_butane_actual):.2f}°C)")
print(f"Neural Network Prediction: {butane_pred_nn:.2f}°C (Error: {abs(butane_pred_nn - y_butane_actual):.2f}°C)")

# Plot the data, fitted lines, and Butane prediction
plt.figure(figsize=(12, 8))

# Plot training data
plt.scatter(X, y, color='blue', label='Training data', zorder=3)

# Sort X for smooth line plotting
X_range = np.linspace(min(X)[0], max(X)[0], 100).reshape(-1, 1)
y_pred_lr_range = lr_model.predict(X_range)
y_pred_nn_range = nn_model.predict(X_range)

plt.plot(X_range, y_pred_lr_range, color='red', linewidth=2, label='Linear Regression', zorder=1)
plt.plot(X_range, y_pred_nn_range, color='green', linewidth=2, label='Neural Network', zorder=2)

# Highlight Butane prediction
plt.scatter(X_butane, y_butane_actual, color='purple', s=100, marker='*', label='Butane (Actual)', zorder=4)
plt.scatter(X_butane, butane_pred_lr, color='orange', s=100, marker='x', label='Butane (LR Prediction)', zorder=4)
plt.scatter(X_butane, butane_pred_nn, color='cyan', s=100, marker='+', label='Butane (NN Prediction)', zorder=4)

# Add vertical lines to show prediction
plt.vlines(x=X_butane[0][0], ymin=min(butane_pred_lr, butane_pred_nn, y_butane_actual)-10,
           ymax=max(butane_pred_lr, butane_pred_nn, y_butane_actual)+10,
           linestyles='dashed', colors='gray')

plt.xlabel('Molecular Weight (MW)')
plt.ylabel('Boiling Point (°C)')
plt.title('Model Comparison with Butane Prediction')
plt.legend()
plt.grid(True)
plt.savefig('butane_prediction_comparison.png')
plt.show()

# Create a zoomed-in plot focusing on the Butane prediction
plt.figure(figsize=(10, 6))
y_min = min(butane_pred_lr, butane_pred_nn, y_butane_actual) - 20
y_max = max(butane_pred_lr, butane_pred_nn, y_butane_actual) + 20
x_min = X_butane[0][0] - 10
x_max = X_butane[0][0] + 10

plt.scatter(X_butane, y_butane_actual, color='purple', s=150, marker='*', label='Butane (Actual)', zorder=4)
plt.scatter(X_butane, butane_pred_lr, color='orange', s=150, marker='x', label='Butane (LR Prediction)', zorder=4)
plt.scatter(X_butane, butane_pred_nn, color='cyan', s=150, marker='+', label='Butane (NN Prediction)', zorder=4)

# Plot the model lines in the zoomed region
X_zoom = np.linspace(x_min, x_max, 50).reshape(-1, 1)
y_pred_lr_zoom = lr_model.predict(X_zoom)
y_pred_nn_zoom = nn_model.predict(X_zoom)

plt.plot(X_zoom, y_pred_lr_zoom, color='red', linewidth=2, label='Linear Regression', zorder=1)
plt.plot(X_zoom, y_pred_nn_zoom, color='green', linewidth=2, label='Neural Network', zorder=2)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Molecular Weight (MW)')
plt.ylabel('Boiling Point (°C)')
plt.title('Zoomed View of Butane Prediction')
plt.legend()
plt.grid(True)
plt.savefig('butane_prediction_zoomed.png')
plt.show()

# Create a bar chart comparing the predictions
plt.figure(figsize=(8, 6))
models = ['Actual', 'Linear Regression', 'Neural Network']
values = [y_butane_actual, butane_pred_lr, butane_pred_nn]
colors = ['purple', 'red', 'green']

plt.bar(models, values, color=colors)
plt.axhline(y=y_butane_actual, color='gray', linestyle='--', label='Actual Value')
plt.ylabel('Boiling Point (°C)')
plt.title('Butane Boiling Point: Actual vs Predicted')
plt.grid(axis='y')

# Add value labels on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + (5 if v >= 0 else -5), f"{v:.2f}°C", ha='center')

plt.savefig('butane_prediction_comparison_bar.png')
plt.show()