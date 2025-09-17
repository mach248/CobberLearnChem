import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Step 1: Generate noisy dataset ---
x = np.linspace(0, 10, 100)
true_m, true_b = 2, 5
true_y = true_m * x + true_b
noise = np.random.normal(0, 2, size=x.shape)
y = true_y + noise
X = x.reshape(-1, 1)

# --- Step 2: Train model with sklearn ---
model = LinearRegression()
model.fit(X, y)
model_m = model.coef_[0]
model_b = model.intercept_

print(f"Model slope: {model_m:.3f}, intercept: {model_b:.3f}")

# --- Step 3: Build loss landscape ---
slopes = np.linspace(0, 4, 100)        # slope values to test
intercepts = np.linspace(0, 10, 100)   # intercept values to test

MSE_grid = np.zeros((len(slopes), len(intercepts)))

for i, m in enumerate(slopes):
    for j, b in enumerate(intercepts):
        guess_y = m * x + b
        MSE_grid[i, j] = mean_squared_error(y, guess_y)

# --- Step 4: Plot heatmap ---
plt.figure(figsize=(8,6))
plt.imshow(MSE_grid.T,
           extent=[slopes.min(), slopes.max(), intercepts.min(), intercepts.max()],
           origin="lower", aspect="auto", cmap="plasma")  # purple=high, yellow=low
plt.colorbar(label="MSE")
plt.xlabel("Slope (m)")
plt.ylabel("Intercept (b)")
plt.title("Loss Landscape (MSE over slope/intercept)")

# Mark true values (⭐) and model’s solution (X)
plt.scatter(true_m, true_b, color="white", edgecolor="black", marker="*", s=200, label="True Line (m=2, b=5)")
plt.scatter(model_m, model_b, color="red", marker="x", s=100, label="Model Solution")

plt.legend()
plt.show()
