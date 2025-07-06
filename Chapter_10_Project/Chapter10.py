import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 20)
noise = np.random.normal(0, 2, size=x.shape)
y = 2 * x + 5 + noise

# Reshape x to a 2D array (required by scikit-learn)
X = x.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(x, y, label='Noisy data')
plt.plot(x, y_pred, color='green', label='Fitted line')
plt.title('Linear Regression on Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('linear_regression.png')
plt.show()


# Print model parameters
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
