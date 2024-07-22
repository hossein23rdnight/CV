import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
import matplotlib.pyplot as plt
from ultralytics import YOLO







#--------------------------------------------------------------------------------------------------

def calculate_accuracy(ground_truths: list, predictions: list) -> dict:
    total_characters = 0
    correct_characters = 0
    correct_plates = 0
    for gt, pred in zip(ground_truths, predictions):
        total_characters += len(gt)
        correct_characters += sum(1 for a, b in zip(gt, pred) if a == b)
        if gt == pred:
            correct_plates += 1
    car = correct_characters / total_characters
    par = correct_plates / len(ground_truths)
    return {"Character Accuracy Rate (CAR)": car, "Plate Accuracy Rate (PAR)": par}

def evaluate_model(image_dir, annotations_dir, input_shape=(64, 288), num_images=500):
    images, labels = load_data(image_dir, annotations_dir, input_shape, num_images)
    predictions = []
    for image in images:
        plate_content = read_plate(image)
        predictions.append(plate_content)
    ground_truths = []
    for label_set in labels:
        ground_truth = ''.join(classes[np.argmax(char_label)] for char_label in label_set)
        ground_truths.append(ground_truth)
    accuracy = calculate_accuracy(ground_truths, predictions)
    print("Model Accuracy:")
    print(f"Character Accuracy Rate (CAR): {accuracy['Character Accuracy Rate (CAR)']:.2f}")
    print(f"Plate Accuracy Rate (PAR): {accuracy['Plate Accuracy Rate (PAR)']:.2f}")

def load_data(image_dir, annotations_dir, input_shape=(64, 288), num_images=500):
    images = []
    labels = [[] for _ in range(num_chars)]
    count = 0
    for annotation_file in os.listdir(annotations_dir):
        if count >= num_images:
            break
        if annotation_file.endswith(".txt"):
            annotation_path = os.path.join(annotations_dir, annotation_file)
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            image_filename = annotation_file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, input_shape[:2][::-1])  
            images.append(image)
            char_data = []
            for line in lines[:8]:  
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                char_data.append((class_id, x_center, y_center, width, height))
            char_data.sort(key=lambda x: x[1])
            for i, (class_id, _, _, _, _) in enumerate(char_data):
                label = to_categorical(class_id, num_classes=num_classes)
                labels[i].append(label)
            count += 1
    images = np.array(images)
    labels = [np.array(label_set) for label_set in labels]
    return images, labels

#--------------------------------------------------------------------------------------------------

















classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "P", "T", "S", "J", "CH", "HE", "KH", 
    "D", "Z", "R", "ZE", "ZH", "SIN", "SHIN", "SAD", 
    "ZAD", "TA", "ZA", "AIN", "GHAIN", "F", "Q", "K", 
    "G", "L", "M", "N", "V", "H", "Y", "IRAN", "Plate"
]

num_classes = len(classes)
num_chars = 8
input_shape = (100, 450, 3)
regression_input_shape = (200, 200, 3)

regression_model_path = "/Users/hossein/Desktop/CV/PROJECTS/cnn_model_sali.h5"
classification_model_path = "/Users/hossein/Desktop/CV/PROJECTS/character_classification_model.h5"

custom_objects = {
    'mse': MeanSquaredError(),
    'mean_absolute_error': MeanAbsoluteError(),
    'mean_squared_error': MeanSquaredError()
}

regression_model = tf.keras.models.load_model(regression_model_path, custom_objects=custom_objects)
classification_model = tf.keras.models.load_model(classification_model_path)

def preprocess_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
  
    image_resized = cv2.resize(image, (target_shape[1], target_shape[0]))  
    image_resized = np.expand_dims(image_resized, axis=0)  
    return image_resized

def corners(image: np.ndarray) -> np.ndarray:
 
    original_size = image.shape[:2]
    image_resized = preprocess_image(image, regression_input_shape[:2])
    image_normalized = image_resized / 255.0
    
    predicted_points = regression_model.predict(image_normalized)[0]
    
    h, w = original_size
    predicted_points[0::2] = predicted_points[0::2] * w
    predicted_points[1::2] = predicted_points[1::2] * h
    
    return predicted_points

def extract_plate(image: np.ndarray, points: np.ndarray) -> np.ndarray:
   
    points = points.reshape(4, 2)
    src = np.array(points, dtype='float32')
    dst = np.array([
        [0, 0],
        [input_shape[1] - 1, 0],
        [input_shape[1] - 1, input_shape[0] - 1],
        [0, input_shape[0] -1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    plate_image = cv2.warpPerspective(image, M, (input_shape[1], input_shape[0]))

    return plate_image

def read_plate(image: np.ndarray) -> str:
  
    preprocessed_image = preprocess_image(image, input_shape[:2])
    predictions = classification_model.predict(preprocessed_image)
    predicted_chars = [classes[np.argmax(pred)] for pred in predictions]
    return ''.join(predicted_chars)

def read_single(image_path: str) -> str:
   
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    plate_points = corners(image)
    
    plate_image = extract_plate(image, plate_points)
    
    plate_content = read_plate(plate_image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Extracted Plate: {plate_content}")
    plt.axis("off")
    
    plt.show()
    
    return plate_content

def read_multiple(image_path: str) -> list:
    
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    yolo_model = YOLO('yolov8n.pt')  
    results = yolo_model(image_path)
    
    plates_content = []
    
    for result in results:
        for bbox in result.boxes:
            confidence = bbox.conf.item()  
            class_id = bbox.cls.item()  
            
            if class_id == 2 and confidence > 0.5:  # Class ID 2 for 'car' in COCO dataset
                x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())
                car_image = image[y_min:y_max, x_min:x_max]
                
                try:
                    plate_points = corners(car_image)
                    plate_image = extract_plate(car_image, plate_points)
                    plate_content = read_plate(plate_image)
                    plates_content.append(plate_content)
                    
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
                    plt.title("Detected Car")
                    plt.axis("off")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Extracted Plate: {plate_content}")
                    plt.axis("off")
                    
                    plt.show()
                except Exception as e:
                    print(f"Error processing car: {e}")
    
    return plates_content


if __name__ == '__main__':
    
    
    
    # image_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs"
    # annotations_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/anns/xmls"
    
    # evaluate_model(image_dir, annotations_dir, input_shape, num_images=500)
    
    
    
#--------------------------------------------------------------------------------------------------


    test_image_path_multi = "/Users/hossein/Desktop/CV/PROJECTS/multiple-cars/day_16856.jpg"
    predicted_plates = read_multiple(test_image_path_multi)
    for i, plate in enumerate(predicted_plates):
        print(f"Predicted License Plate {i + 1}: {plate}")


    # test_image_path_single = "/Users/hossein/Desktop/CV/PROJECTS/four-corners/images/1.jpg"
    # predicted_plate = read_single(test_image_path_single)
    
    # images_folder = "/Users/hossein/Desktop/CV/PROJECTS/four-corners/images"
    
#--------------------------------------------------------------------------------------------------

# for filename in os.listdir(images_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(images_folder, filename)
#         predicted_plate = read_single(image_path)
#         image = cv2.imread(image_path)
#         cv2.putText(image, predicted_plate, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('Predicted Plate', image)
        
#         cv2.waitKey(0)  

# cv2.destroyAllWindows()