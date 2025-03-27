import keras
import numpy as np
from PIL import Image

# Load the model using TFSMLayer
model = keras.layers.TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict(image):
    # Load and preprocess the image
    if isinstance(image, str):
        img = Image.open(image)  # Resize based on model input
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model(img_array)  # Call the model
    predictions = predictions.numpy()  # Convert Tensor to NumPy array
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    return labels[class_idx], confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predicted_class, confidence = predict(image_path)
        print(f"Predicted: {predicted_class} with {confidence:.2f} confidence")