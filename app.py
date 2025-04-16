import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Loadtrained model
model = tf.keras.models.load_model("isl_model.h5")

IMG_SIZE = 128

dataset_path = "dataset" 
if os.path.exists(dataset_path):
    CLASS_LABELS = sorted(os.listdir(dataset_path))
else:
    CLASS_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] 

print("Loaded Class Labels:", CLASS_LABELS)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = Image.fromarray(img)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))

 
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    if predicted_class < len(CLASS_LABELS):
        predicted_letter = CLASS_LABELS[predicted_class]
    else:
        predicted_letter = "Unknown"

    # Display result
    cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    cv2.imshow("ISL Gesture Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
