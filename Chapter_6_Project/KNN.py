import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Select features (drop things like Name, Ticket, Cabin, etc.)
df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Now you can proceed with your simulation:
df_test = df_encoded.copy()
known_ages = df_test['Age'].dropna()
sample_indices = known_ages.sample(frac=0.2, random_state=42).index

# Store true values
true_values = df_test.loc[sample_indices, 'Age'].copy()

# Set values to NaN for simulation
df_test.loc[sample_indices, 'Age'] = np.nan

# Impute using KNN
imputer = KNNImputer(n_neighbors=5)
df_test_imputed = pd.DataFrame(imputer.fit_transform(df_test), columns=df_test.columns)

# Compare results
predicted_values = df_test_imputed.loc[sample_indices, 'Age']
mae = mean_absolute_error(true_values, predicted_values)
print(f"\nMean Absolute Error of KNN Imputation: {mae:.2f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, alpha=0.6)
plt.plot([0, 80], [0, 80], color='red', linestyle='--')
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs KNN-Predicted Age")
plt.show()
