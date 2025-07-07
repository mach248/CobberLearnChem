import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Step 1: Create a 5x5 image with a plus sign
image = np.array([[0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0]])

# Step 2: Define a horizontal line detector filter
horizontal_filter = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]])

# Step 3: Apply the convolution
feature_map = convolve2d(image, horizontal_filter, mode='same')

# Step 4: Print and visualize
print("Feature Map:")
print(feature_map)

plt.imshow(feature_map, cmap='gray_r', interpolation='nearest')
plt.colorbar(label='Filter Response')
plt.title("Feature Map from Horizontal Filter")
plt.axis('off')
plt.savefig("feature_map.png", bbox_inches='tight', dpi=300)
plt.show()
