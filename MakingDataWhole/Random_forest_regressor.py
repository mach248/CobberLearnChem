# Import necessary libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os

# Create a directory to save plots if it doesn't exist
os.makedirs('titanic_imputation_comparison', exist_ok=True)

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Display the first few rows
print("First few rows of the Titanic dataset:")
print(titanic_data.head())

# Check for missing values in the Age column
missing_age_count = titanic_data['age'].isna().sum()
print(f"\nNumber of missing values in the Age column: {missing_age_count}")

# Calculate the mean of known ages
original_age_mean = titanic_data['age'].mean()
print(f"Mean age of passengers with known ages: {original_age_mean:.2f}")

# Prepare data for imputation
# First, we need to convert categorical variables to numeric
titanic_numeric = pd.get_dummies(titanic_data,
                                 columns=['sex', 'embarked', 'class', 'deck', 'embark_town', 'alive', 'alone'],
                                 drop_first=True)

# Select features for imputation (excluding 'age' and non-predictive columns)
features_for_imputation = titanic_numeric.drop(['age', 'who', 'adult_male'], axis=1)

# Standardize the features (important for KNN and regression)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_imputation)
features_scaled_df = pd.DataFrame(features_scaled, columns=features_for_imputation.columns)

# Add age back to the scaled features
features_scaled_df['age'] = titanic_data['age']

# Split data into rows with known age and rows with missing age
known_age = features_scaled_df[~features_scaled_df['age'].isna()].copy()
missing_age = features_scaled_df[features_scaled_df['age'].isna()].copy()

# Split known age data into train and test sets
X_train, X_test = train_test_split(known_age, test_size=0.3, random_state=42)

# Create a copy of X_test with some age values artificially set to NaN for testing
X_test_with_nan = X_test.copy()
# Save the true age values before introducing NaNs
y_true = X_test_with_nan['age'].copy()

# Artificially introduce NaN values in 30% of the test set
np.random.seed(42)
mask = np.random.rand(len(X_test_with_nan)) < 0.3
X_test_with_nan.loc[mask, 'age'] = np.nan
print(f"Number of artificially introduced NaN values in test set: {mask.sum()}")

# Prepare data for all models
X_train_features = X_train.drop('age', axis=1)
y_train = X_train['age']
X_test_features = X_test.drop('age', axis=1)
X_test_nan_features = X_test_with_nan.drop('age', axis=1)

# Dictionary to store results
imputation_results = {
    'Method': [],
    'MAE': [],
    'Predicted': [],
    'Actual': []
}

# ===================== 1. KNN IMPUTATION =====================
print("\n========== KNN IMPUTATION ==========")

# Apply KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputer.fit(X_train)

# Predict ages for test set to evaluate model
X_test_imputed_knn = knn_imputer.transform(X_test_with_nan)
X_test_imputed_knn_df = pd.DataFrame(X_test_imputed_knn, columns=X_test_with_nan.columns, index=X_test_with_nan.index)
y_pred_knn = X_test_imputed_knn_df['age']

# Calculate Mean Absolute Error only for the artificially introduced NaNs
mae_knn = mean_absolute_error(y_true[mask], y_pred_knn[mask])
print(f"KNN Imputation MAE: {mae_knn:.2f} years")

# Store results
imputation_results['Method'].append('KNN')
imputation_results['MAE'].append(mae_knn)
imputation_results['Predicted'].append(y_pred_knn[mask].values)
imputation_results['Actual'].append(y_true[mask].values)

# Plot actual vs predicted ages for KNN
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred_knn[mask], alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Age')
plt.ylabel('KNN-Predicted Age')
plt.title(f'KNN Imputation: Actual vs Predicted Ages (MAE: {mae_knn:.2f} years)')
plt.grid(True)
plt.savefig('titanic_imputation_comparison/knn_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 2. LOG-LINEAR REGRESSION IMPUTATION =====================
print("\n========== LOG-LINEAR REGRESSION IMPUTATION ==========")

# Apply log transformation to age (adding a small constant to handle zeros)
log_y_train = np.log1p(y_train)  # log(1+x) to handle zeros

# Train a linear regression model on log-transformed data
lr_model = LinearRegression()
lr_model.fit(X_train_features, log_y_train)

# Predict log-ages for test set
log_y_pred_lr = lr_model.predict(X_test_nan_features)

# Transform back to original scale
y_pred_lr = np.expm1(log_y_pred_lr)  # exp(x)-1 to reverse log1p

# Create a copy of the test set with imputed values
X_test_imputed_lr = X_test_with_nan.copy()
X_test_imputed_lr.loc[mask, 'age'] = y_pred_lr[mask]

# Calculate Mean Absolute Error only for the artificially introduced NaNs
mae_lr = mean_absolute_error(y_true[mask], y_pred_lr[mask])
print(f"Log-Linear Regression Imputation MAE: {mae_lr:.2f} years")

# Store results
imputation_results['Method'].append('Log-Linear Regression')
imputation_results['MAE'].append(mae_lr)
imputation_results['Predicted'].append(y_pred_lr[mask])
imputation_results['Actual'].append(y_true[mask].values)

# Plot actual vs predicted ages for Log-Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred_lr[mask], alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Age')
plt.ylabel('Log-Linear Regression Predicted Age')
plt.title(f'Log-Linear Regression Imputation: Actual vs Predicted Ages (MAE: {mae_lr:.2f} years)')
plt.grid(True)
plt.savefig('titanic_imputation_comparison/log_linear_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 3. RANDOM FOREST IMPUTATION =====================
print("\n========== RANDOM FOREST IMPUTATION ==========")

# Train a Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train_features, y_train)

# Predict ages for test set
y_pred_rf = rf_model.predict(X_test_nan_features)

# Create a copy of the test set with imputed values
X_test_imputed_rf = X_test_with_nan.copy()
X_test_imputed_rf.loc[mask, 'age'] = y_pred_rf[mask]

# Calculate Mean Absolute Error only for the artificially introduced NaNs
mae_rf = mean_absolute_error(y_true[mask], y_pred_rf[mask])
print(f"Random Forest Imputation MAE: {mae_rf:.2f} years")

# Store results
imputation_results['Method'].append('Random Forest')
imputation_results['MAE'].append(mae_rf)
imputation_results['Predicted'].append(y_pred_rf[mask])
imputation_results['Actual'].append(y_true[mask].values)

# Plot actual vs predicted ages for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_true[mask], y_pred_rf[mask], alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Age')
plt.ylabel('Random Forest Predicted Age')
plt.title(f'Random Forest Imputation: Actual vs Predicted Ages (MAE: {mae_rf:.2f} years)')
plt.grid(True)
plt.savefig('titanic_imputation_comparison/random_forest_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== COMPARISON OF ALL METHODS =====================
print("\n========== COMPARISON OF ALL IMPUTATION METHODS ==========")

# Create a combined scatter plot of all methods
plt.figure(figsize=(15, 8))

# Plot each method with different colors and markers
plt.scatter(imputation_results['Actual'][0], imputation_results['Predicted'][0],
            alpha=0.6, color='blue', label=f'KNN (MAE: {mae_knn:.2f})')
plt.scatter(imputation_results['Actual'][1], imputation_results['Predicted'][1],
            alpha=0.6, color='green', label=f'Log-Linear (MAE: {mae_lr:.2f})')
plt.scatter(imputation_results['Actual'][2], imputation_results['Predicted'][2],
            alpha=0.6, color='red', label=f'Random Forest (MAE: {mae_rf:.2f})')

# Add the ideal prediction line
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='Ideal Prediction')

plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Comparison of Imputation Methods: Actual vs Predicted Ages')
plt.grid(True)
plt.legend()
plt.savefig('titanic_imputation_comparison/all_methods_comparison_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a bar chart comparing MAE values
methods = imputation_results['Method']
mae_values = imputation_results['MAE']

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, mae_values, color=['blue', 'green', 'red'])

# Add MAE values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Imputation Method')
plt.ylabel('Mean Absolute Error (years)')
plt.title('Comparison of Imputation Methods: Mean Absolute Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('titanic_imputation_comparison/mae_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a box plot to compare the distribution of errors for each method
error_data = []
error_labels = []

for i, method in enumerate(methods):
    # Calculate absolute errors
    abs_errors = np.abs(np.array(imputation_results['Actual'][i]) - np.array(imputation_results['Predicted'][i]))
    error_data.append(abs_errors)
    error_labels.append(method)

plt.figure(figsize=(10, 6))
plt.boxplot(error_data, labels=error_labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'))
plt.ylabel('Absolute Error (years)')
plt.title('Distribution of Absolute Errors by Imputation Method')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('titanic_imputation_comparison/error_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply the best method to the entire dataset and save the results
print("\n========== APPLYING BEST IMPUTATION METHOD TO ENTIRE DATASET ==========")

# Determine the best method based on MAE
best_method_index = np.argmin(imputation_results['MAE'])
best_method = imputation_results['Method'][best_method_index]
print(f"Best imputation method: {best_method} (MAE: {imputation_results['MAE'][best_method_index]:.2f})")

# Apply the best method to the entire dataset
if best_method == 'KNN':
    # Prepare full dataset
    full_data_features = features_scaled_df.copy()

    # Apply KNN imputation
    full_data_imputed = knn_imputer.transform(full_data_features)
    full_data_imputed_df = pd.DataFrame(full_data_imputed, columns=full_data_features.columns,
                                        index=full_data_features.index)

    # Add the imputed ages back to the original dataset
    titanic_data['age_imputed'] = full_data_imputed_df['age']

elif best_method == 'Log-Linear Regression':
    # Prepare features for the entire dataset
    full_data_features = features_scaled_df.drop('age', axis=1)

    # Predict log-ages for all missing values
    log_y_pred_full = lr_model.predict(full_data_features)

    # Transform back to original scale
    y_pred_full = np.expm1(log_y_pred_full)

    # Create a new column with original ages where available and imputed ages where missing
    titanic_data['age_imputed'] = titanic_data['age'].copy()
    titanic_data.loc[titanic_data['age'].isna(), 'age_imputed'] = y_pred_full[titanic_data['age'].isna()]

else:  # Random Forest
    # Prepare features for the entire dataset
    full_data_features = features_scaled_df.drop('age', axis=1)

    # Predict ages for all missing values
    y_pred_full = rf_model.predict(full_data_features)

    # Create a new column with original ages where available and imputed ages where missing
    titanic_data['age_imputed'] = titanic_data['age'].copy()
    titanic_data.loc[titanic_data['age'].isna(), 'age_imputed'] = y_pred_full[titanic_data['age'].isna()]

# Report average ages before and after imputation
print(f"Average age before imputation (excluding missing values): {original_age_mean:.2f}")
print(f"Average age after {best_method} imputation (all values): {titanic_data['age_imputed'].mean():.2f}")
print(f"Average of imputed values only: {titanic_data.loc[titanic_data['age'].isna(), 'age_imputed'].mean():.2f}")

# Visualize the distribution of ages before and after imputation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(titanic_data['age'].dropna(), kde=True, color='blue')
plt.axvline(original_age_mean, color='red', linestyle='--', label=f'Mean: {original_age_mean:.2f}')
plt.title('Original Age Distribution (Missing Values Excluded)')
plt.xlabel('Age')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(titanic_data['age_imputed'], kde=True, color='green')
plt.axvline(titanic_data['age_imputed'].mean(), color='red', linestyle='--',
            label=f'Mean: {titanic_data["age_imputed"].mean():.2f}')
plt.title(f'Age Distribution After {best_method} Imputation')
plt.xlabel('Age')
plt.legend()

plt.tight_layout()
plt.savefig('titanic_imputation_comparison/final_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the final dataset with imputed values
titanic_data.to_csv('titanic_imputation_comparison/titanic_with_imputed_age.csv', index=False)

print(f"\nAll plots and data have been saved to the 'titanic_imputation_comparison' directory.")