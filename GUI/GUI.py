import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QLabel, QWidget, QFrame,
                             QLineEdit, QGridLayout, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

# Set matplotlib backend before importing FigureCanvas
import matplotlib

matplotlib.use('QtAgg')

# Now import the canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Custom matplotlib canvas for PyQt6 compatibility"""

    def __init__(self, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class LinearRegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNA/RNA Size Calculator")
        self.setGeometry(100, 100, 900, 700)

        # Initialize variables
        self.csv_file_path = None
        self.data = None
        self.model = None
        self.slope = None
        self.intercept = None

        # Create the main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create file selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select CSV File")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        main_layout.addLayout(file_layout)

        # Create fit button
        self.fit_button = QPushButton("Fit Standard Curve")
        self.fit_button.clicked.connect(self.perform_regression)
        self.fit_button.setEnabled(False)  # Disable until file is selected
        main_layout.addWidget(self.fit_button)

        # Create results section
        results_group = QGroupBox("Regression Results")
        results_layout = QGridLayout()

        # Slope display
        results_layout.addWidget(QLabel("Slope:"), 0, 0)
        self.slope_value = QLabel("---")
        self.slope_value.setFrameShape(QFrame.Shape.Panel)
        self.slope_value.setFrameShadow(QFrame.Shadow.Sunken)
        self.slope_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.slope_value, 0, 1)

        # Intercept display
        results_layout.addWidget(QLabel("Intercept:"), 0, 2)
        self.intercept_value = QLabel("---")
        self.intercept_value.setFrameShape(QFrame.Shape.Panel)
        self.intercept_value.setFrameShadow(QFrame.Shadow.Sunken)
        self.intercept_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.intercept_value, 0, 3)

        # R² display
        results_layout.addWidget(QLabel("R² Value:"), 0, 4)
        self.r2_value = QLabel("---")
        self.r2_value.setFrameShape(QFrame.Shape.Panel)
        self.r2_value.setFrameShadow(QFrame.Shadow.Sunken)
        self.r2_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.r2_value, 0, 5)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # Create size calculator section
        calc_group = QGroupBox("Size Calculator")
        calc_layout = QGridLayout()

        # Distance input
        calc_layout.addWidget(QLabel("Enter Distance (cm):"), 0, 0)
        self.distance_input = QLineEdit()
        self.distance_input.setValidator(QDoubleValidator(0.0, 1000.0, 2))  # Only allow numbers
        self.distance_input.setEnabled(False)  # Disable until regression is performed
        calc_layout.addWidget(self.distance_input, 0, 1)

        # Calculate button
        self.calc_button = QPushButton("Calculate Size")
        self.calc_button.clicked.connect(self.calculate_size)
        self.calc_button.setEnabled(False)  # Disable until regression is performed
        calc_layout.addWidget(self.calc_button, 0, 2)

        # Size output
        calc_layout.addWidget(QLabel("Size (BP):"), 0, 3)
        self.size_output = QLabel("---")
        self.size_output.setFrameShape(QFrame.Shape.Panel)
        self.size_output.setFrameShadow(QFrame.Shadow.Sunken)
        self.size_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_output.setMinimumWidth(100)
        calc_layout.addWidget(self.size_output, 0, 4)

        calc_group.setLayout(calc_layout)
        main_layout.addWidget(calc_group)

        # Create matplotlib plot area
        plot_group = QGroupBox("Standard Curve")
        plot_layout = QVBoxLayout()
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)

        # Set the main widget
        self.setCentralWidget(main_widget)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.csv_file_path = file_path
            self.file_label.setText(f"Selected: {file_path}")
            self.fit_button.setEnabled(True)

            # Clear previous results
            self.slope_value.setText("---")
            self.intercept_value.setText("---")
            self.r2_value.setText("---")
            self.size_output.setText("---")
            self.distance_input.setEnabled(False)
            self.calc_button.setEnabled(False)
            self.canvas.axes.clear()
            self.canvas.draw()

    def perform_regression(self):
        try:
            # Load the CSV file
            self.data = pd.read_csv(self.csv_file_path)

            # Check if the CSV has at least two columns
            if len(self.data.columns) < 2:
                raise ValueError("CSV file must have at least two columns")

            # Extract the first two columns
            x_column = self.data.columns[0]  # Distance in cm
            y_column = self.data.columns[1]  # log BP

            X = self.data[x_column].values.reshape(-1, 1)  # Reshape for sklearn
            y = self.data[y_column].values

            # Perform linear regression
            self.model = LinearRegression()
            self.model.fit(X, y)

            # Get slope and intercept
            self.slope = self.model.coef_[0]
            self.intercept = self.model.intercept_

            # Calculate R² value
            y_pred = self.model.predict(X)
            r2 = r2_score(y, y_pred)

            # Update the labels
            self.slope_value.setText(f"{self.slope:.4f}")
            self.intercept_value.setText(f"{self.intercept:.4f}")
            self.r2_value.setText(f"{r2:.4f}")

            # Create the plot
            self.canvas.axes.clear()
            self.canvas.axes.scatter(X, y, color='blue', label='Data points')
            self.canvas.axes.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
            self.canvas.axes.set_title('DNA/RNA Standard Curve')
            self.canvas.axes.set_xlabel('Distance (cm)')
            self.canvas.axes.set_ylabel('log BP')
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)

            # Update the canvas
            self.canvas.draw()

            # Enable the size calculator
            self.distance_input.setEnabled(True)
            self.calc_button.setEnabled(True)

        except FileNotFoundError:
            self.file_label.setText(f"Error: File not found")
        except ValueError as e:
            self.file_label.setText(f"Error: {e}")
        except Exception as e:
            self.file_label.setText(f"An unexpected error occurred: {e}")

    def calculate_size(self):
        try:
            # Get the distance value
            distance = float(self.distance_input.text())

            # Calculate the log BP using the regression model
            log_bp = self.slope * distance + self.intercept

            # Convert log BP to actual BP
            bp = 10 ** log_bp

            # Update the size output
            self.size_output.setText(f"{int(bp)}")

            # Highlight the point on the plot
            self.update_plot_with_point(distance, log_bp)

        except ValueError:
            self.size_output.setText("Invalid input")
        except Exception as e:
            self.size_output.setText(f"Error: {str(e)}")

    def update_plot_with_point(self, distance, log_bp):
        # Get the current plot
        ax = self.canvas.axes

        # Remove previous highlighted point if it exists
        for artist in ax.get_children():
            if hasattr(artist, '_highlight_point') and artist._highlight_point:
                artist.remove()

        # Add the new point
        highlight = ax.scatter([distance], [log_bp], color='green', s=100,
                               edgecolor='black', zorder=10)
        highlight._highlight_point = True

        # Update the canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearRegressionApp()
    window.show()
    sys.exit(app.exec())