import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "P", "T", "S", "J", "CH", "HE", "KH", 
    "D", "Z", "R", "ZE", "ZH", "SIN", "SHIN", "SAD", 
    "ZAD", "TA", "ZA", "AIN", "GHAIN", "F", "Q", "K", 
    "G", "L", "M", "N", "V", "H", "Y", "IRAN", "Plate"
]

test_image_paths = [
    "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs/00000.jpg",
    "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs/00001.jpg",
    "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs/00002.jpg"
]

def predict_and_display(image_paths, model, classes):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (200, 200)) 
        image_input = np.expand_dims(image_resized, axis=0)
        
        predictions = model.predict(image_input)
        print(f"Predictions shape: {predictions.shape}")
        
        max_indices = [np.argmax(pred) for pred in predictions[0]]
        print(f"Max indices: {max_indices}")

        if any(idx >= len(classes) for idx in max_indices):
            raise ValueError("Predicted index is out of range of the classes list")

        predicted_chars = [classes[idx] for idx in max_indices]
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {''.join(predicted_chars)}")
        plt.axis('off')
        plt.show()

model = load_model("/Users/hossein/Desktop/CV/PROJECTS/mainPlate.keras")

model.summary()

predict_and_display(test_image_paths, model, classes)
