import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
import matplotlib.pyplot as plt

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

def check_points_inside_image(points, height, width):
 
    return np.all(points[:, 0] >= 0) and np.all(points[:, 0] < width) and np.all(points[:, 1] >= 0) and np.all(points[:, 1] < height)



def shift_images_and_labels(images, labels, max_shift_fraction=0.4, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    shifted_images = []
    shifted_labels = []

    for image, label in zip(images, labels):
        h, w = image.shape[:2]

        shift_x = int(random.uniform(-max_shift_fraction, max_shift_fraction) * w)
        shift_y = int(random.uniform(-max_shift_fraction, max_shift_fraction) * h)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        shifted_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        points = np.array(label).reshape(-1, 2)
        shifted_points = points + np.array([shift_x, shift_y])

        if check_points_inside_image(shifted_points, h, w):
            shifted_images.append(shifted_image)
            shifted_labels.append(shifted_points.flatten().tolist())
        else:
            print("Some points are outside the image boundaries. Image and labels are discarded.")


    return np.array(shifted_images), np.array(shifted_labels)




def normalize_labels(labels, target_size=(200, 200)):
    h, w = target_size
    normalized_labels = np.copy(labels)
    normalized_labels[:, 0::2] = normalized_labels[:, 0::2] / w  
    normalized_labels[:, 1::2] = normalized_labels[:, 1::2] / h 
    return normalized_labels

def create_cnn_model(input_shape=(200, 200, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(8, activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def show_images_with_points(images, labels):
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.figure(figsize=(8, 8))  
        img_uint8 = (image * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        x_points = label[0::2]
        y_points = label[1::2]
        plt.scatter(x_points, y_points, color='red', s=100, marker='o', edgecolors='black')  
        corners = ['UL', 'UR', 'LR', 'LL']
        for j, (x, y) in enumerate(zip(x_points, y_points)):
            plt.text(x, y, f'{corners[j]} ({x:.2f}, {y:.2f})', color='yellow', fontsize=12)
        plt.title(f'Image {i+1}')
        plt.axis('off')
        plt.show()

def denormalize_points(points, target_size=(200, 200)):
    h, w = target_size
    denormalized_points = np.copy(points)
    denormalized_points[0::2] = denormalized_points[0::2] * w  
    denormalized_points[1::2] = denormalized_points[1::2] * h  
    return denormalized_points

def show_images_with_points_predict(images, actual_points, predicted_points, target_size=(200, 200)):
    for i, (image, actual, predicted) in enumerate(zip(images, actual_points, predicted_points)):
        plt.figure(figsize=(8, 8)) 
        img_uint8 = (image * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        actual_denorm = denormalize_points(actual, target_size)
        predicted_denorm = denormalize_points(predicted, target_size)
        x_actual = actual_denorm[0::2]
        y_actual = actual_denorm[1::2]
        plt.scatter(x_actual, y_actual, color='red', s=100, marker='o', edgecolors='black', label='Actual')
        x_pred = predicted_denorm[0::2]
        y_pred = predicted_denorm[1::2]
        plt.scatter(x_pred, y_pred, color='blue', s=100, marker='x', label='Predicted')
        corners = ['UL', 'UR', 'LR', 'LL']
        for j, (x, y) in enumerate(zip(x_actual, y_actual)):
            plt.text(x, y, f'{corners[j]} ({x:.2f}, {y:.2f})', color='yellow', fontsize=12)
        plt.title(f'Image {i+1}')
        plt.axis('off')
        plt.legend()
        plt.show()

image_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/images'
label_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/labels'

images, labels = read_and_normalize_data(image_folder, label_folder)

shift_images, shift_labels = shift_images_and_labels(images, labels)

all_images = np.concatenate([images, shift_images], axis=0)
all_labels = np.concatenate([labels, shift_labels], axis=0)

all_labels = normalize_labels(all_labels, target_size=(200, 200))

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

cnn_model = create_cnn_model()

cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='mse',
                   metrics=[MeanAbsoluteError(), MeanSquaredError()])
cnn_model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

loss, mae, mse = cnn_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test mae: {mae}")
print(f"Test mse: {mse}")

model_path = '/Users/hossein/Desktop/CV/PROJECTS/cnn_model_sali.h5'
cnn_model.save(model_path)
print(f"Model saved at {model_path}")

y_test_pred = cnn_model.predict(X_test[:10])
show_images_with_points_predict(X_test[:10], y_test[:10], y_test_pred)

def corners(image: np.ndarray) -> np.ndarray:
    in_size = (200, 200)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, in_size)
    image_normalized = image_resized / 255.0
    feed_model = np.expand_dims(image_normalized, axis=0)
    
    predicted_points = cnn_model.predict(feed_model)[0]
    
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

mse = eval(X_test, y_test, cnn_model)
print(f"Mean Squared Error on the test set: {mse}")
