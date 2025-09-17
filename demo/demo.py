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

# --- Step 3: Loss landscape (MSE grid) ---
slopes = np.linspace(0, 4, 100)
intercepts = np.linspace(0, 10, 100)

MSE_grid = np.zeros((len(slopes), len(intercepts)))
for i, m in enumerate(slopes):
    for j, b in enumerate(intercepts):
        guess_y = m * x + b
        MSE_grid[i, j] = mean_squared_error(y, guess_y)

# --- Step 4: Gradient Descent Implementation ---
def compute_gradients(m, b, x, y):
    y_pred = m * x + b
    error = y_pred - y
    dm = (2/len(x)) * np.sum(error * x)
    db = (2/len(x)) * np.sum(error)
    return dm, db

# Initialize parameters randomly
m, b = np.random.uniform(0, 4), np.random.uniform(0, 10)
lr = 0.01   # learning rate
steps = 200 # total iterations

path = []
for step in range(steps):
    dm, db = compute_gradients(m, b, x, y)
    m -= lr * dm
    b -= lr * db
    path.append((m, b))

path = np.array(path)

# --- Step 5: Plot loss landscape + path ---
plt.figure(figsize=(8,6))
plt.imshow(MSE_grid.T,
           extent=[slopes.min(), slopes.max(), intercepts.min(), intercepts.max()],
           origin="lower", aspect="auto", cmap="viridis_r")
plt.colorbar(label="MSE")
plt.xlabel("Slope (m)")
plt.ylabel("Intercept (b)")
plt.title("Loss Landscape with Gradient Descent Path")

# Mark true line (⭐) and sklearn solution (X)
plt.scatter(true_m, true_b, color="white", edgecolor="black", marker="*", s=200, label="True Line (m=2, b=5)")
plt.scatter(model_m, model_b, color="red", marker="x", s=100, label="Model Solution")

# Plot full path in gray
plt.plot(path[:,0], path[:,1], color="gray", linewidth=1, alpha=0.6, label="GD Path (full)")

# Highlight last 20 steps in red
plt.plot(path[-20:,0], path[-20:,1], color="red", marker="o", markersize=4, linewidth=2, label="GD Path (last 20)")

plt.legend()
plt.show()
