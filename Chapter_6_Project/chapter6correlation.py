import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset (update the path if needed)
df = sns.load_dataset("titanic")

# Step 1: Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

print(df.columns)

# Step 2: Generate correlation matrix
corr_matrix = numeric_df.corr()

# Step 3: Display correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Step 4: Show top correlations with age
print("Top correlations with 'age':")
print(corr_matrix['age'].sort_values(ascending=False))

