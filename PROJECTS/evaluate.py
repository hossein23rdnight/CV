import numpy as np
from sklearn.metrics import mean_squared_error
import cv2
import os

import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.model_selection import train_test_split



#----------------------------------------------------------

def read_and_normalize_data(image_folder, label_folder, target_size=(200, 200)):
    images = []
    labels = []

    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    for image_file in image_files:
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = image / 255.0  

            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(label_folder, label_file)
            if label_file in label_files:
                with open(label_path, 'r') as file:
                    label_data = file.read().strip().split()
                    label_data = list(map(float, label_data[1:])) 
          


                    h, w = target_size
                    label_data[0::2] = [x * w for x in label_data[0::2]]  
                    label_data[1::2] = [y * h for y in label_data[1::2]]  

                    labels.append(label_data)
                    images.append(image)

    return np.array(images), np.array(labels)

image_folder = '/Users/hossein/Desktop/CV/PROJECTS/presentation/images'
label_folder = '/Users/hossein/Desktop/CV/PROJECTS/presentation/labels'

#----------------------------------------------------------
def normalize_labels(labels, target_size=(200, 200)):
    h, w = target_size
    normalized_labels = np.copy(labels)
    normalized_labels[:, 0::2] = normalized_labels[:, 0::2] / w  
    normalized_labels[:, 1::2] = normalized_labels[:, 1::2] / h  
    return normalized_labels






custom_objects = {
    'mse': MeanSquaredError(),
    'mean_absolute_error': MeanAbsoluteError(),
    'mean_squared_error': MeanSquaredError()
}

regression_model_path = "/Users/hossein/Desktop/CV/PROJECTS/cnn_model_sali.h5"
regression_model = tf.keras.models.load_model(regression_model_path, custom_objects=custom_objects)



def corners(image: np.ndarray) -> np.ndarray:
    in_size = (200, 200)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, in_size)
    image_normalized = image_resized / 255.0
    feed_model = np.expand_dims(image_normalized, axis=0)
    
    predicted_points = regression_model.predict(feed_model)[0]
    
    h, w = original_size
    predicted_points[0::2] = predicted_points[0::2] * w
    predicted_points[1::2] = predicted_points[1::2] * h
    
    return predicted_points

def eval(images, labels, model, target_size=(200, 200)):
    M = []
    for image, actual_points in zip(images, labels):
        predicted_points = corners(image)
        mse = np.mean((predicted_points - actual_points) ** 2)
        M.append(mse)
    
    return np.mean(M) 





images, labels = read_and_normalize_data(image_folder, label_folder)
all_images=images
all_labels=labels

all_labels = normalize_labels(all_labels)
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.4, random_state=42)

mse = eval(X_test, y_test, regression_model)
print(f"Mean Squared Error on the test set: {mse}")
