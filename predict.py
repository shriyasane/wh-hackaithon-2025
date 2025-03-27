import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# Load the model
model = tf.keras.models.load_model("saved_model")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict(image_path):
    # Load and preprocess image
    img = Image.open(image_path).resize((224, 224))  # Adjust size if needed
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    return labels[class_idx], confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predicted_class, confidence = predict(image_path)
        print(f"Predicted: {predicted_class} with {confidence:.2f} confidence")
