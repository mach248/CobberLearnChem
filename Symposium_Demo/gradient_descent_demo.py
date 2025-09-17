import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate noisy dataset
np.random.seed(42)
x = np.linspace(0, 10, 50)
true_y = 2 * x + 5
noise = np.random.normal(0, 2, size=x.shape)
y = true_y + noise
X = x.reshape(-1, 1)

# Fit the "best" line with LinearRegression for reference
model = LinearRegression()
model.fit(X, y)
best_slope = model.coef_[0]
best_intercept = model.intercept_
print(f"Best-fit line: y = {best_slope:.2f}x + {best_intercept:.2f}")

# --- Create loss landscape (MSE over slope & intercept grid) ---
slopes = np.linspace(0, 4, 100)       # range of slopes to test
intercepts = np.linspace(0, 10, 100)  # range of intercepts to test
mse_values = np.zeros((len(intercepts), len(slopes)))

for i, b in enumerate(intercepts):
    for j, m in enumerate(slopes):
        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)
        mse_values[i, j] = mse

# --- Plot heatmap of loss landscape ---
plt.figure(figsize=(8, 6))
plt.imshow(mse_values, origin='lower',
           extent=[slopes[0], slopes[-1], intercepts[0], intercepts[-1]],
           aspect='auto', cmap='plasma')
plt.colorbar(label="MSE (loss)")
plt.scatter(best_slope, best_intercept, color="white", marker="x", s=100, label="Best fit")
plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("Loss Landscape (MSE across slope & intercept)")
plt.legend()
plt.savefig("heatmap.png", bbox_inches='tight', dpi=300)
plt.show()
