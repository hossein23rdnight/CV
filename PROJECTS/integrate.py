import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.utils import to_categorical


classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "P", "T", "S", "J", "CH", "HE", "KH", 
    "D", "Z", "R", "ZE", "ZH", "SIN", "SHIN", "SAD", 
    "ZAD", "TA", "ZA", "AIN", "GHAIN", "F", "Q", "K", 
    "G", "L", "M", "N", "V", "H", "Y", "IRAN", "Plate"
]

num_classes = len(classes)
num_chars = 8
input_shape = (200, 200, 3)

model = tf.keras.models.load_model("character_classification_model_MAIN.h5")

def preprocess_image(image: np.ndarray) -> np.ndarray:

    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))  
    image_resized = np.expand_dims(image_resized, axis=0) 
    return image_resized






def load_data(image_dir, annotations_dir, input_shape=(64, 288)):
    images = []
    labels = [[] for _ in range(num_chars)]

    for annotation_file in os.listdir(annotations_dir):
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
    
    images = np.array(images)
    labels = [np.array(label_set) for label_set in labels]
    
    return images, labels



def read_plate(image: np.ndarray) -> str:
  
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_chars = [classes[np.argmax(pred)] for pred in predictions]
    return ''.join(predicted_chars)

def character_level_accuracy(y_true, y_pred):
 
    correct = 0
    total = 0
    for true_chars, pred_chars in zip(y_true, y_pred):
        for true, pred in zip(true_chars, pred_chars):
            if np.argmax(true) == np.argmax(pred):
                correct += 1
            total += 1
    return correct / total

def plate_level_accuracy(y_true, y_pred):
  
    correct = 0
    for true_plate, pred_plate in zip(y_true, y_pred):
        if true_plate == pred_plate:
            correct += 1
    return correct / len(y_true)

def evaluate_model(val_images: np.ndarray, val_labels: list[np.ndarray]) -> dict:
  
    val_loss, *val_accuracies = model.evaluate(val_images, val_labels)
    average_accuracy = np.mean(val_accuracies)
    
    predictions = model.predict(val_images)
    char_acc = character_level_accuracy(val_labels, predictions)
    
    true_plates = [''.join(classes[np.argmax(val_labels[char][i])] for char in range(num_chars)) for i in range(len(val_images))]
    predicted_plates = [''.join(classes[np.argmax(predictions[char][i])] for char in range(num_chars)) for i in range(len(val_images))]
    
    plate_acc = plate_level_accuracy(true_plates, predicted_plates)
    
    return {
        'average_accuracy': average_accuracy,
        'character_level_accuracy': char_acc,
        'plate_level_accuracy': plate_acc
    }

if __name__ == '__main__':
    image_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs"
    annotations_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/anns/xmls"
    
  
    
    images, labels = load_data(image_dir, annotations_dir, input_shape[:2])
    split_index = int(0.8 * len(images))
    val_images, val_labels = images[split_index:], [label_set[split_index:] for label_set in labels]
    
    metrics = evaluate_model(val_images, val_labels)
    print(f"Validation Average Accuracy: {metrics['average_accuracy'] * 100:.2f}%")
    print(f"Character-Level Accuracy: {metrics['character_level_accuracy'] * 100:.2f}%")
    print(f"Plate-Level Accuracy: {metrics['plate_level_accuracy'] * 100:.2f}%")
    
    test_image_path = "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs/00000.jpg"
    test_image = cv2.imread(test_image_path)
    predicted_plate = read_plate(test_image)
    print(f"Predicted License Plate: {predicted_plate}")
