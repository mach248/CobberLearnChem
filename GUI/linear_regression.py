import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def perform_linear_regression(csv_file):
    try:
        # Load the CSV file
        print(f"Loading data from {csv_file}...")
        data = pd.read_csv(csv_file)

        # Check if the CSV has at least two columns
        if len(data.columns) < 2:
            raise ValueError("CSV file must have at least two columns")

        # Extract the first two columns
        x_column = data.columns[0]
        y_column = data.columns[1]

        print(f"Using '{x_column}' as X and '{y_column}' as Y")

        X = data[x_column].values.reshape(-1, 1)  # Reshape for sklearn
        y = data[y_column].values

        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Get slope and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        # Print results
        print("\nLinear Regression Results:")
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")

        # Create predictions for plotting
        y_pred = model.predict(X)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
        plt.title('Linear Regression')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig('regression_plot.png')
        print("Plot saved as 'regression_plot.png'")

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Get CSV file path from user
    csv_file = input("Enter the path to your CSV file: ")
    perform_linear_regression(csv_file)