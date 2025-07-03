import numpy as np

# Actual and predicted values
actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
predicted = np.array([1, 2, 4, 3, 5, 7, 6, 8, 10, 9])


print("Actual:", actual)
print("Predicted:", predicted)

# Calculate residuals
residuals = actual - predicted
print("Residuals:", residuals)

# Define evaluation metrics function
def evaluate_predictions(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mae, mse, r2

# Calculate and print metrics
mae, mse, r2 = evaluate_predictions(actual, predicted)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Actual and predicted values
actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
predicted = np.array([1, 2, 4, 3, 5, 7, 6, 8, 10, 9])

# Residuals
residuals = actual - predicted
print("Residuals:", residuals)

# Manual evaluation
def evaluate_predictions(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mae, mse, r2

mae_manual, mse_manual, r2_manual = evaluate_predictions(actual, predicted)

print("\nManual calculation:")
print(f"MAE: {mae_manual:.2f}")
print(f"MSE: {mse_manual:.2f}")
print(f"R²: {r2_manual:.2f}")

# Using sklearn
mae_sklearn = mean_absolute_error(actual, predicted)
mse_sklearn = mean_squared_error(actual, predicted)

print("\nsklearn.metrics:")
print(f"MAE: {mae_sklearn:.2f}")
print(f"MSE: {mse_sklearn:.2f}")
