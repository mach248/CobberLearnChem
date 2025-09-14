import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class CaliforniaHousingPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("California Housing Price Predictor")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        # Set style
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('TEntry', font=('Arial', 12))

        # Load and train the model
        self.load_and_train_model()

        # Create GUI elements
        self.create_widgets()

    def load_and_train_model(self):
        # Show loading message
        loading_label = ttk.Label(self.root, text="Loading and training model, please wait...", font=('Arial', 14))
        loading_label.pack(pady=20)
        self.root.update()

        try:
            # Load California housing dataset
            housing = fetch_california_housing(as_frame=True)
            X = housing.data
            y = housing.target

            # Store feature names for later use
            self.feature_names = X.columns.tolist()

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Remove loading label
            loading_label.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or train model: {str(e)}")
            self.root.quit()

    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="California Housing Price Predictor", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Input fields
        self.entries = {}

        # Only include the features we want to expose in the GUI
        gui_features = [
            ("MedInc", "Median Income", 8.3, 0.5, 15.0),
            ("HouseAge", "House Age (years)", 28.0, 1.0, 50.0),
            ("AveRooms", "Average Rooms", 5.4, 1.0, 10.0),
            ("AveBedrms", "Average Bedrooms", 1.1, 0.5, 5.0),
            ("Population", "Population", 1425.0, 100.0, 10000.0),
            ("AveOccup", "Average Occupancy", 3.1, 1.0, 10.0)
        ]

        # Create input fields with labels and default values
        for i, (feature, label, default, min_val, max_val) in enumerate(gui_features):
            row = i + 1

            # Label
            ttk.Label(main_frame, text=f"{label}:").grid(row=row, column=0, sticky=tk.W, pady=5)

            # Entry with validation
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(main_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, pady=5)

            # Store entry and its variable
            self.entries[feature] = (var, entry, min_val, max_val)

        # Predict button
        predict_button = ttk.Button(main_frame, text="Predict House Value", command=self.predict)
        predict_button.grid(row=len(gui_features) + 1, column=0, columnspan=2, pady=20)

        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=10)
        result_frame.grid(row=len(gui_features) + 2, column=0, columnspan=2, sticky=tk.EW)

        # Result label
        self.result_var = tk.StringVar(value="$0")
        result_label = ttk.Label(result_frame, textvariable=self.result_var, font=('Arial', 16, 'bold'))
        result_label.pack(pady=10)

        # Add a note about the missing features
        note_text = ("Note: Latitude and Longitude are set to default values.\n"
                     "This model works best for central California locations.")
        note_label = ttk.Label(main_frame, text=note_text, font=('Arial', 10, 'italic'))
        note_label.grid(row=len(gui_features) + 3, column=0, columnspan=2, pady=(10, 0))

    def validate_inputs(self):
        """Validate all inputs are within acceptable ranges"""
        for feature, (var, entry, min_val, max_val) in self.entries.items():
            try:
                value = var.get()
                if value < min_val or value > max_val:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"{feature} should be between {min_val} and {max_val}."
                    )
                    entry.focus()
                    return False
            except:
                messagebox.showwarning("Invalid Input", f"Please enter a valid number for {feature}.")
                entry.focus()
                return False
        return True

    def predict(self):
        """Make prediction based on input values"""
        if not self.validate_inputs():
            return

        try:
            # Create a dictionary to hold all feature values
            input_data = {}

            # Get values from entries
            for feature, (var, _, _, _) in self.entries.items():
                input_data[feature] = var.get()

            # Add default values for latitude and longitude (central California)
            input_data['Latitude'] = 37.0
            input_data['Longitude'] = -120.0

            # Create DataFrame with all features in the correct order
            df = pd.DataFrame([input_data])

            # Ensure columns are in the right order
            df = df.reindex(columns=self.feature_names)

            # Make prediction
            prediction = self.model.predict(df)[0]

            # Convert to dollars (the target is in $100,000s)
            price = prediction * 100000

            # Update result
            self.result_var.set(f"${price:,.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CaliforniaHousingPredictorGUI(root)
    root.mainloop()