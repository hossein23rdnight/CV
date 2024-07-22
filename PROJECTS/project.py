import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
import cv2
import matplotlib.pyplot as plt
import numpy as np





def read_and_normalize_data(image_folder, label_folder, target_size=(256, 256)):
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

image_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/images'
label_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/labels'



def normalize_labels(labels, target_size=(256, 256)):
    h, w = target_size
    normalized_labels = np.copy(labels)
    normalized_labels[:, 0::2] = normalized_labels[:, 0::2] / w  
    normalized_labels[:, 1::2] = normalized_labels[:, 1::2] / h 
    return normalized_labels






def build_model():
    model = tf.keras.Sequential([
        Input(shape=(256, 256, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='sigmoid')  
    ])

    return model







#------------------------------------------FUNCTION

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


def rotate_images_and_labels(images, labels):
    rotated_images = []
    rotated_labels = []

    for image, label in zip(images, labels):
        angle = random.uniform(0, 180)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        points = np.array(label).reshape(-1, 2)
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])

        rotated_points = rotation_matrix.dot(points_ones.T).T

        rotated_images.append(rotated_image)
        rotated_labels.append(rotated_points.flatten())

    return np.array(rotated_images), np.array(rotated_labels)


def blur_images_and_labels(images, labels, max_kernel_size=5):
    blurred_images = []
    blurred_labels = []

    for image, label in zip(images, labels):

        kernel_size = random.randint(3, max_kernel_size) * 2 - 1
        if kernel_size <= 0:
            kernel_size = 3  

        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        blurred_images.append(blurred_image)

        blurred_labels.append(label)

    return np.array(blurred_images), np.array(blurred_labels)




def adjust_contrast_images_and_labels(images, labels):
    adjusted_images = []
    adjusted_labels = []

    for image, label in zip(images, labels):

        contrast_factor = random.uniform(0.5, 2.0)
        beta = random.randint(-20, 20)

        img_255 = (image * 255).astype(np.uint8)

        adjusted_image_255 = cv2.convertScaleAbs(img_255, alpha=contrast_factor, beta=beta)

        adjusted_image = adjusted_image_255 / 255.0

        adjusted_images.append(adjusted_image)

        adjusted_labels.append(label)

    return np.array(adjusted_images), np.array(adjusted_labels)


def check_points_inside_image(points, height, width):
 
    return np.all(points[:, 0] >= 0) and np.all(points[:, 0] < width) and np.all(points[:, 1] >= 0) and np.all(points[:, 1] < height)



def perspective_transform_images_and_labels(images, labels, margin=0.3):
    transformed_images = []
    transformed_labels = []

    for image, label in zip(images, labels):
        h, w = image.shape[:2]

        perspective_matrix = np.array([
            [1 + np.random.uniform(-margin, margin), np.random.uniform(-margin, margin), np.random.uniform(-margin, margin) * w],
            [np.random.uniform(-margin, margin), 1 + np.random.uniform(-margin, margin), np.random.uniform(-margin, margin) * h],
            [np.random.uniform(-margin, margin) * 1e-4, np.random.uniform(-margin, margin) * 1e-4, 1]
        ])

        transformed_image = cv2.warpPerspective(image, perspective_matrix, (w, h))

        points = np.array(label).reshape(-1, 2)
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])

        transformed_points_homogeneous = perspective_matrix.dot(points_homogeneous.T).T
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2, np.newaxis]

        if check_points_inside_image(transformed_points, h, w):
            transformed_images.append(transformed_image)
            transformed_labels.append(transformed_points.flatten())
        else:
            print("Some points are outside the image boundaries. Image and labels are discarded.")

    return np.array(transformed_images), np.array(transformed_labels)



def crop_images_and_labels(images, labels, min_crop_margin=0.05, max_crop_margin=0.25, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed) 

    cropped_images = []
    cropped_labels = []

    for image, label in zip(images, labels):
        h, w = image.shape[:2]

        crop_margin = random.uniform(min_crop_margin, max_crop_margin)

        crop_x = int(crop_margin * w)
        crop_y = int(crop_margin * h)
        new_w = w - 2 * crop_x
        new_h = h - 2 * crop_y

        
        cropped_image = image[crop_y:crop_y + new_h, crop_x:crop_x + new_w]

        padded_image = np.zeros_like(image)
        padded_image[crop_y:crop_y + new_h, crop_x:crop_x + new_w] = cropped_image

        points = np.array(label).reshape(-1, 2)
        cropped_points = points - np.array([crop_x, crop_y])

        if check_points_inside_image(cropped_points, new_h, new_w):
            cropped_images.append(padded_image)
            cropped_labels.append(label)  
        else:
            print("Some points are outside the image boundaries. Image and labels are discarded.")

    return np.array(cropped_images), np.array(cropped_labels)



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



def denormalize_points(points, target_size=(256, 256)):
    h, w = target_size
    denormalized_points = np.copy(points)
    denormalized_points[0::2] = denormalized_points[0::2] * w  
    denormalized_points[1::2] = denormalized_points[1::2] * h  
    return denormalized_points

def normalize_labels(labels, target_size=(256, 256)):
    h, w = target_size
    normalized_labels = np.copy(labels)
    normalized_labels[:, 0::2] = normalized_labels[:, 0::2] / w  
    normalized_labels[:, 1::2] = normalized_labels[:, 1::2] / h  
    return normalized_labels

def show_images_with_points_predict(images, actual_points, predicted_points, target_size=(256, 256)):
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


#-------------------------------------------------

images, labels = read_and_normalize_data(image_folder, label_folder)

# rotated_images, rotated_labels = rotate_images_and_labels(images, labels)
# blurred_images, blurred_labels = blur_images_and_labels(images, labels, max_kernel_size=3)
# contrast_images,contrast_labels = adjust_contrast_images_and_labels(images, labels)
#transformed_images,transformed_labels = perspective_transform_images_and_labels(images, labels)
# crop_images,crop_labels = crop_images_and_labels(images, labels)
shift_images,shift_labels = shift_images_and_labels(images, labels)


# all_images = np.concatenate([images, rotated_images, blurred_images, contrast_images, transformed_images, crop_images, shift_images], axis=0)
# all_labels = np.concatenate([labels, rotated_labels, blurred_labels, contrast_labels, transformed_labels, crop_labels, shift_labels], axis=0)


# all_images=images
# all_labels=labels
all_images = np.concatenate([images,  shift_images], axis=0)
all_labels = np.concatenate([labels, shift_labels], axis=0)

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(all_images.shape)
print(all_labels.shape)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

#show_images_with_points(all_images, all_labels)


all_labels = normalize_labels(all_labels)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

yolo_model = build_model()




yolo_model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='mse',
                   metrics=[MeanAbsoluteError(), MeanSquaredError()])
yolo_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

loss, mae, mse = yolo_model.evaluate(X_test, y_test)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(f"Test Loss: {loss}")
print(f"Test mae: {mae}")
print(f"Test mse: {mse}")
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


# Save the model
model_path = '/Users/hossein/Desktop/CV/PROJECTS/yolo_model.h5'
yolo_model.save(model_path)
print(f"Model saved at {model_path}")

# y_test_pred = yolo_model.predict(X_test)
# show_images_with_points_predict(X_test, y_test, y_test_pred)

y_test_pred = yolo_model.predict(X_test[:10])

show_images_with_points_predict(X_test[:10], y_test[:10], y_test_pred)


def corners(image: np.ndarray) -> np.ndarray:
    in_size = (256, 256)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, in_size)
    image_normalized = image_resized / 255.0
    feed_model = np.expand_dims(image_normalized, axis=0)
    
    predicted_points = yolo_model.predict(feed_model)[0]
    
    h, w = original_size
    predicted_points[0::2] = predicted_points[0::2] * w
    predicted_points[1::2] = predicted_points[1::2] * h
    
    return predicted_points

def eval(images, labels, model, target_size=(256, 256)):
    M = []
    for image, actual_points in zip(images, labels):
        predicted_points = corners(image)
        mse = np.mean((predicted_points - actual_points) ** 2)
        M.append(mse)
    
    return np.mean(M) 

mse = eval(X_test, y_test, yolo_model)
print(f"Mean Squared Error on the test set: {mse}")

