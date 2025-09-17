import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# 1. Generate the 'x' and noisy 'y' datasets
x = np.linspace(0, 10, 50)
true_slope = 2
true_intercept = 5
y_true = true_slope * x + true_intercept
noise = np.random.normal(0, 2, x.shape)
y_noisy = y_true + noise

# Reshape the data for scikit-learn
X_reshaped = x.reshape(-1, 1)

# Train the Linear Regression model to find the optimal solution
model = LinearRegression()
model.fit(X_reshaped, y_noisy)
model_slope = model.coef_[0]
model_intercept = model.intercept_
y_pred = model.predict(X_reshaped)

# --- FIRST PLOT: Model Prediction vs. True Line (Unchanged) ---
print("Generating and saving the Model Prediction plot...")
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, label='Noisy Data', color='blue')
plt.plot(x, y_true, color='red', linestyle='--', linewidth=2, label='True Line')
plt.plot(x, y_pred, color='green', linewidth=2, label='Model Prediction')
plt.title('Linear Regression on Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('model_prediction_plot.png')
print("Saved 'model_prediction_plot.png'")
plt.close()

# --- SECOND PLOT: Loss Landscape Heatmap with Realistic Path ---
print("Generating and saving the Loss Landscape plot with a realistic path...")

# 2. Define the grid of slope and intercept values to test
slopes_to_test = np.linspace(0, 4, 50)
intercepts_to_test = np.linspace(3, 7, 50)

# Calculate the MSE for each combination
mse_values = np.zeros((len(slopes_to_test), len(intercepts_to_test)))
for i, slope in enumerate(slopes_to_test):
    for j, intercept in enumerate(intercepts_to_test):
        y_predicted_grid = slope * x + intercept
        mse = mean_squared_error(y_noisy, y_predicted_grid)
        mse_values[i, j] = mse

# 3. Simulate a more realistic, "zig-zagging" path
# We'll manually define a path of points that descends the loss landscape
# A real algorithm would compute these steps based on the gradient.
path_slopes = np.array([0.5, 0.8, 1.2, 1.5, 1.9, 2.1, 2.05, model_slope])
path_intercepts = np.array([6.5, 6.2, 5.8, 5.5, 5.2, 5.0, 4.95, model_intercept])

# 4. Plot the heatmap
plt.figure(figsize=(12, 8))
plt.imshow(
    mse_values,
    extent=[slopes_to_test.min(), slopes_to_test.max(), intercepts_to_test.min(), intercepts_to_test.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis_r'
)

# 5. Add markers and the path
plt.plot(path_slopes, path_intercepts, color='red', linestyle='-', marker='o',
         markersize=8, label='Simulated Realistic Path')

# Plot the true solution (yellow star)
plt.plot(true_slope, true_intercept, 'y*', markersize=20, label='True Solution')

# Plot the model's learned solution (red X)
plt.plot(model_slope, model_intercept, 'rX', markersize=20, label="Model's Solution")

# Add plot labels and title
plt.title('Loss Landscape with a Simulated Realistic Optimization Path')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.colorbar(label='Mean Squared Error (MSE)')
plt.legend()

# Save the plot
plt.savefig('loss_landscape_with_path.png')
print("Saved 'loss_landscape_with_path.png'")
plt.close()

print("\nAll tasks completed. Your project directory contains the two saved plots.")