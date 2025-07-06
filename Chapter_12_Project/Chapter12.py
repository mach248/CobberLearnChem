import pandas as pd
from sklearn.neural_network import MLPRegressor

# Step 1: Create the dataset
data = {
    'Compound': ['Water', 'Methane', 'Ethanol', 'Propane'],
    'MW': [18, 16, 46, 44],
    'BP': [100, -161, 78, -42]  # Boiling points in °C
}
df = pd.DataFrame(data)

# Step 2: Set features and target
X = df[['MW']]
y = df['BP']

# Step 3: Create and train the MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(15, 7), max_iter=5000, random_state=42)
mlp.fit(X, y)

# Step 4: Predict the boiling point of Butane (MW = 58)
butane_mw = pd.DataFrame([[58]], columns=['MW'])
predicted_bp = mlp.predict(butane_mw)

# Step 5: Print result
print(f"Predicted boiling point of Butane (MW=58): {predicted_bp[0]:.2f} °C")
