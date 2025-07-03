import pandas as pd
import seaborn as sns

# Load Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Show first few rows
print(df.head())

missing_age = df['age'].isnull().sum()
print(f"Missing values in Age column: {missing_age}")

# Step 2: Calculate the mean of 'age'
mean_age = df['age'].mean()

## Step 3: Fill missing values with the mean
df['age'] = df['age'].fillna(mean_age)    # ✅ Safe and future-proof


# Step 4: Check again for missing values
print(f"Missing values after imputation: {df['age'].isnull().sum()}")

