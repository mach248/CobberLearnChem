import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Define the data
data = {
    'MW': [180, 250, 80, 300, 150, 400, 90, 200],
    'HBD': [5, 2, 1, 1, 4, 3, 0, 2],
    'HBA': [6, 3, 2, 2, 5, 4, 1, 3],
    'Soluble': [1, 0, 1, 0, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['MW', 'HBD', 'HBA']]
y = df['Soluble']

# Create and train the classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(10, 5))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['Not Soluble', 'Soluble'],
          filled=True,
          rounded=True,
          fontsize=10)  # Adjust font size here
plt.title("Decision Tree for Solubility Prediction", fontsize=12)
plt.tight_layout()
plt.savefig("solubility_tree_smaller_boxes.png")
plt.show()