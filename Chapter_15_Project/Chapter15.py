from sklearn.datasets import fetch_california_housing

# Load as pandas DataFrame for convenience
housing = fetch_california_housing(as_frame=True)

X = housing.data        # Features (8 columns)
y = housing.target      # Target (median house value)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("\nFirst 5 rows:\n", X.head())
print("\nTarget preview:\n", y.head())

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# List of models
models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Support Vector Regressor", SVR())
]

# Train and evaluate
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"{name} R² Score: {score:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Combine features and target into one DataFrame
df = pd.concat([housing.data, housing.target.rename("Target")], axis=1)

# Set up Seaborn styling
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Plot feature distributions
df.hist(bins=30, figsize=(15, 10), edgecolor="black")
plt.suptitle("California Housing Feature Distributions", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# Save the figure before displaying
plt.savefig("feature_distributions.png", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# Save the figure before displaying
plt.savefig("heat_map.png", bbox_inches='tight', dpi=300)
plt.show()

best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

import pandas as pd

# Get the correct feature names from training data
feature_names = X.columns

# New house data
new_data = pd.DataFrame([[
    5.0, 30.0, 6.0, 1.0, 300.0, 2.5, 34.5, -119.5
]], columns=feature_names)

# Predict
prediction = best_model.predict(new_data)
print(f"Predicted house value: ${prediction[0] * 100000:.2f}")


