import cv2
import numpy as np
from keras.models import load_model
#from Emailer import send_notification


def classify_plant_health(image_path):
    # Load the trained model
    model = load_model("plant_health_classifier.h5")

    # Read and preprocess the input image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match training size
    img = img / 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(np.expand_dims(img, axis=0))

    # Interpret the prediction
    if prediction[0][1] > 0.5:
#        send_notification()
        return "Unhealthy plant"
    else:
        return "Healthy plant"

# Example usage:
input_image_path = "TestImages\images.jpg"
result = classify_plant_health(input_image_path)
print(f"Result: {result}")