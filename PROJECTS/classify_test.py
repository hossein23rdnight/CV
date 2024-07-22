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
        image_resized = cv2.resize(image, (450, 100)) 
        image_input = np.expand_dims(image_resized, axis=0)
        
        predictions = model.predict(image_input)
        predicted_chars = [classes[np.argmax(pred)] for pred in predictions]
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {''.join(predicted_chars)}")
        plt.axis('off')
        plt.show()

model = load_model("/Users/hossein/Desktop/CV/PROJECTS/character_classification_model.h5")

predict_and_display(test_image_paths, model, classes)
