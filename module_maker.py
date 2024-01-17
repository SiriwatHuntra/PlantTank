import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Set paths to your dataset
healthy_dir = "healthyPlant"
unhealthy_dir = "unhealthyPlant"

# Initialize lists for images and labels
images = []
labels = []

# Load healthy plant images
for img_name in os.listdir(healthy_dir):
    img_path = os.path.join(healthy_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to a common size
    images.append(img)
    labels.append(0)  # Healthy label (0)

# Load unhealthy plant images
for img_name in os.listdir(unhealthy_dir):
    img_path = os.path.join(unhealthy_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to a common size
    images.append(img)
    labels.append(1)  # Unhealthy label (1)

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess images (normalize pixel values)
X_train = X_train / 255.0
X_val = X_val / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Two output classes: healthy and unhealthy

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Export the trained model
model.save("plant_health_classifier.h5")

print("Training completed! Model saved as 'plant_health_classifier.h5'.")
