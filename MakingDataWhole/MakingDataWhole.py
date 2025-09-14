# Import necessary libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Display the first ten rows
print("First 10 rows of the Titanic dataset:")
print(titanic_data.head(10))

# Check for missing values in the Age column
missing_age_count = titanic_data['age'].isna().sum()
print(f"\nNumber of missing values in the Age column: {missing_age_count}")

# Calculate the mean of known ages
original_age_mean = titanic_data['age'].mean()
print(f"Mean age of passengers with known ages: {original_age_mean:.2f}")

# Prepare data for KNN imputation
# First, we need to convert categorical variables to numeric
titanic_numeric = pd.get_dummies(titanic_data, columns=['sex', 'embarked', 'class', 'deck', 'embark_town', 'alive', 'alone'], drop_first=True)

# Select features for imputation (excluding 'age' and non-predictive columns)
features_for_imputation = titanic_numeric.drop(['age', 'who', 'adult_male'], axis=1)

# Standardize the features (important for KNN)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_imputation)
features_scaled_df = pd.DataFrame(features_scaled, columns=features_for_imputation.columns)

# Add age back to the scaled features
features_scaled_df['age'] = titanic_data['age']

# Split data into rows with known age and rows with missing age
known_age = features_scaled_df[~features_scaled_df['age'].isna()].copy()
missing_age = features_scaled_df[features_scaled_df['age'].isna()].copy()

# Split known age data into train and test sets to evaluate KNN imputation
X_train, X_test = train_test_split(known_age, test_size=0.2, random_state=42)

# Create a copy of X_test with some age values artificially set to NaN for testing
X_test_with_nan = X_test.copy()
# Save the true age values before introducing NaNs
y_true = X_test_with_nan['age'].copy()

# Artificially introduce NaN values in 30% of the test set
np.random.seed(42)
mask = np.random.rand(len(X_test_with_nan)) < 0.3
X_test_with_nan.loc[mask, 'age'] = np.nan
print(f"Number of artificially introduced NaN values in test set: {mask.sum()}")

# Apply KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputer.fit(X_train)

# Predict ages for test set to evaluate model
X_test_imputed = knn_imputer.transform(X_test_with_nan)
X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test_with_nan.columns, index=X_test_with_nan.index)
y_pred = X_test_imputed_df['age']

# Calculate Mean Absolute Error only for the artificially introduced NaNs
mae = mean_absolute_error(y_true[mask], y_pred[mask])
print(f"\nMean Absolute Error (MAE) of KNN imputation model: {mae:.2f} years")

# Plot actual vs predicted ages for the artificially introduced NaNs
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred[mask], alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Age')
plt.ylabel('KNN-Predicted Age')
plt.title('Actual vs KNN-Predicted Ages (for artificially missing values)')
plt.grid(True)
plt.savefig('Actual_v_Predicted_Ages.png', dpi=300, bbox_inches='tight')
plt.show()

# Now apply KNN imputation to the entire dataset
full_data_imputed = knn_imputer.transform(features_scaled_df)
full_data_imputed_df = pd.DataFrame(full_data_imputed, columns=features_scaled_df.columns, index=features_scaled_df.index)

# Add the imputed ages back to the original dataset
titanic_data['age_knn_imputed'] = full_data_imputed_df['age']

# Report average ages before and after imputation
print(f"\nAverage age before imputation (excluding missing values): {original_age_mean:.2f}")
print(f"Average age after KNN imputation (all values): {titanic_data['age_knn_imputed'].mean():.2f}")
print(f"Average of imputed values only: {titanic_data.loc[titanic_data['age'].isna(), 'age_knn_imputed'].mean():.2f}")

# Visualize the distribution of ages before and after imputation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(titanic_data['age'].dropna(), kde=True, color='blue')
plt.axvline(original_age_mean, color='red', linestyle='--', label=f'Mean: {original_age_mean:.2f}')
plt.title('Original Age Distribution (Missing Values Excluded)')
plt.xlabel('Age')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(titanic_data['age_knn_imputed'], kde=True, color='green')
plt.axvline(titanic_data['age_knn_imputed'].mean(), color='red', linestyle='--',
           label=f'Mean: {titanic_data["age_knn_imputed"].mean():.2f}')
plt.title('Age Distribution After KNN Imputation')
plt.xlabel('Age')
plt.legend()

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare imputed values with mean imputation
titanic_data['age_mean_imputed'] = titanic_data['age'].fillna(original_age_mean)

# Plot comparison of imputation methods for missing values only
missing_age_rows = titanic_data[titanic_data['age'].isna()]
plt.figure(figsize=(10, 6))
plt.scatter(range(len(missing_age_rows)), missing_age_rows['age_mean_imputed'],
           label='Mean Imputation', alpha=0.5, color='blue')
plt.scatter(range(len(missing_age_rows)), missing_age_rows['age_knn_imputed'],
           label='KNN Imputation', alpha=0.5, color='green')
plt.axhline(original_age_mean, color='blue', linestyle='--',
           label=f'Mean Age: {original_age_mean:.2f}')
plt.axhline(missing_age_rows['age_knn_imputed'].mean(), color='green', linestyle='--',
           label=f'KNN Imputed Mean: {missing_age_rows["age_knn_imputed"].mean():.2f}')
plt.title('Comparison of Imputation Methods for Missing Age Values')
plt.xlabel('Index of Missing Value')
plt.ylabel('Imputed Age')
plt.legend()
plt.grid(True)
plt.savefig('imputation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()