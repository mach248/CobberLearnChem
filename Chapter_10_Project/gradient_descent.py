import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 20)
noise = np.random.normal(0, 2, size=x.shape)
y = 2 * x + 5 + noise

# Reshape x to 2D (required by scikit-learn)
X = x.reshape(-1, 1)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f"Best-fit Intercept: {model.intercept_:.2f}")
print(f"Best-fit Slope: {model.coef_[0]:.2f}")


# --- Function to compute MSE for custom line ---
def test_line(slope, intercept):
    y_guess = slope * x + intercept
    mse = mean_squared_error(y, y_guess)
    print(f"For slope={slope:.2f}, intercept={intercept:.2f}, MSE={mse:.2f}")
    return mse


# --- Create Loss Landscape ---
slope_vals = np.linspace(0, 4, 100)       # range of slopes to try
intercept_vals = np.linspace(0, 10, 100)  # range of intercepts to try

MSE_grid = np.zeros((len(intercept_vals), len(slope_vals)))

for i, b in enumerate(intercept_vals):
    for j, m in enumerate(slope_vals):
        y_guess = m * x + b
        MSE_grid[i, j] = mean_squared_error(y, y_guess)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    MSE_grid,
    extent=[slope_vals.min(), slope_vals.max(), intercept_vals.min(), intercept_vals.max()],
    origin='lower',
    aspect='auto',
    cmap='plasma'  # low = yellow, high = purple
)
plt.colorbar(label="Mean Squared Error")
plt.scatter(model.coef_[0], model.intercept_, color="white", edgecolor="black", s=80, label="Best Fit")
plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("Loss Landscape (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_landscape.png")
plt.show()
