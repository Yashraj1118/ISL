import os
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET_PATH = r"..\isldata\ISL_Dataset"
CATEGORIES = sorted(os.listdir(DATASET_PATH))  # Ensure this points to a valid directory
NUM_CLASSES = len(CATEGORIES)

IMG_SIZE = 128

# Load dataset
X, Y = [], []

for category in CATEGORIES:
    class_index = CATEGORIES.index(category)
    category_path = os.path.join(DATASET_PATH, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
        X.append(img)
        Y.append(class_index)

# Convert to numpy arrays
X = np.array(X) / 255.0  # Normalize pixel values
Y = to_categorical(Y, NUM_CLASSES)  # One-hot encoding

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Dataset Loaded: {len(X_train)} training images, {len(X_val)} validation images")

# Build CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer
])

# Compile 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val), batch_size=32)

#save
model.save("isl_model.h5")

print("Model trained and saved successfully!")
