import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Path to your image file - you'll need to replace this with your actual path
img_path = 'elephant.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
processed_img = preprocess_input(img_array)

# Make predictions
predictions = model.predict(processed_img)

# Decode and print the top 5 predictions
decoded_predictions = decode_predictions(predictions, top=5)[0]
print("Top 5 predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# Optional: Display the image
plt.imshow(image.load_img(img_path))
plt.axis('off')
plt.show()