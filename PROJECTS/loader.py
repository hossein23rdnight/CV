
import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np



def read_and_normalize_data(image_folder, label_folder, target_size=(256, 256)):
    images = []
    labels = []

    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    for image_file in image_files[:5]:
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
                    # Denormalize 
                    h, w = target_size
                    label_data[0::2] = [x * w for x in label_data[0::2]]  # x coordinates
                    label_data[1::2] = [y * h for y in label_data[1::2]]  # y coordinates

                    labels.append(label_data)
                    images.append(image)

    return np.array(images), np.array(labels)

image_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/images'
label_folder = '/Users/hossein/Desktop/CV/PROJECTS/four-corners/labels'


















#------------------------------------------function
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

        alpha = random.uniform(0.5, 2.0)
        beta = random.randint(-20, 20)

        img_255 = (image * 255).astype(np.uint8)

        adjusted_image_255 = cv2.convertScaleAbs(img_255, alpha=alpha, beta=beta)

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

        #  perspective transformation matrix
        perspective_matrix = np.array([
            [1 + np.random.uniform(-margin, margin), np.random.uniform(-margin, margin), np.random.uniform(-margin, margin) * w],
            [np.random.uniform(-margin, margin), 1 + np.random.uniform(-margin, margin), np.random.uniform(-margin, margin) * h],
            [np.random.uniform(-margin, margin) * 1e-4, np.random.uniform(-margin, margin) * 1e-4, 1]
        ])

        transformed_image = cv2.warpPerspective(image, perspective_matrix, (w, h))

        # Transform the label points
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

        # Transform the label points
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

        # Adjust label points
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



#------------------------------------------------------


images, labels = read_and_normalize_data(image_folder, label_folder)
show_images_with_points(images, labels)



rotated_images, rotated_labels = rotate_images_and_labels(images, labels)
show_images_with_points(rotated_images, rotated_labels)

blurred_images, blurred_labels = blur_images_and_labels(images, labels, max_kernel_size=3)
show_images_with_points(blurred_images, blurred_labels)

contrast_images,contrast_labels = adjust_contrast_images_and_labels(images, labels)
show_images_with_points(contrast_images, contrast_labels)

transformed_images,transformed_labels = perspective_transform_images_and_labels(images, labels)
show_images_with_points(transformed_images, transformed_labels)

crop_images,crop_labels = crop_images_and_labels(images, labels)
show_images_with_points(crop_images,crop_labels)

shift_images,shift_labels = shift_images_and_labels(images, labels)
show_images_with_points(shift_images, shift_labels)




# images, labels = read_and_normalize_data(image_folder, label_folder)

# augmentations = [
#     rotate_images_and_labels,
#     blur_images_and_labels,
#     adjust_contrast_images_and_labels,
#     perspective_transform_images_and_labels,
#     crop_images_and_labels,
#     shift_images_and_labels
# ]

# random.shuffle(augmentations)
# selected_augmentations = augmentations[:max(2, random.randint(2, len(augmentations)))]

# augmented_images_list = [images]
# augmented_labels_list = [labels]

# for augmentation in selected_augmentations:
#     augmented_images, augmented_labels = augmentation(images, labels)
#     augmented_images_list.append(augmented_images)
#     augmented_labels_list.append(augmented_labels)

# all_images = np.concatenate(augmented_images_list, axis=0)
# all_labels = np.concatenate(augmented_labels_list, axis=0)



#----------------------------------------------------GENERATOR


#  on colab
