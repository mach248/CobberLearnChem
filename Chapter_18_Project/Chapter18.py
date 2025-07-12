import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained VGG16 model (with ImageNet weights)
model = VGG16(weights='imagenet')

# Load and preprocess an example image
img_path = 'elephant.jpg'  # Replace with your image file path
img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects 224x224 images
x = image.img_to_array(img)  # Convert to numpy array
x = np.expand_dims(x, axis=0)  # Add batch dimension
x = preprocess_input(x)  # Preprocess the image for VGG16

# Make predictions
predictions = model.predict(x)

# Decode and display top-5 predictions
decoded_predictions = decode_predictions(predictions, top=5)[0]
for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({confidence:.2f})")

import matplotlib.pyplot as plt

plt.imshow(img)
plt.title(f"Prediction: {decoded_predictions[0][1]} ({decoded_predictions[0][2]:.2f})")
plt.axis('off')
plt.savefig("prediction.png", bbox_inches='tight', dpi=300)
plt.show()
