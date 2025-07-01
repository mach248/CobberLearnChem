import matplotlib.pyplot as plt

# Number of carbons for the first 10 linear alkanes
carbons = list(range(1, 11))

# Boiling points in Celsius
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.0]

# Create scatter plot
plt.scatter(carbons, boiling_points, color='blue')

# Add title and labels
plt.title('Boiling Point vs Number of Carbons in Linear Alkanes')
plt.xlabel('Number of Carbons')
plt.ylabel('Boiling Point (°C)')

# Add grid
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('alkane_boiling_points.png', dpi=300)

# Show plot
plt.show()
