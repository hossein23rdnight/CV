import os
import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0



classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "P", "T", "S", "J", "CH", "HE", "KH", 
    "D", "Z", "R", "ZE", "ZH", "SIN", "SHIN", "SAD", 
    "ZAD", "TA", "ZA", "AIN", "GHAIN", "F", "Q", "K", 
    "G", "L", "M", "N", "V", "H", "Y", "IRAN", "Plate"
]

num_classes = len(classes)  
num_chars = 8  
aspect_ratio = 4.5 

#--------------------------------------------------------------------------------------------------
def create_model(input_shape=(64, 288, 3), num_chars=8, num_classes=42):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    
    x = Dense(1024, activation='relu')(x)
    
    outputs = [Dense(num_classes, activation='softmax', name=f'char_{i+1}')(x) for i in range(num_chars)]
    
    model = Model(inputs=input_layer, outputs=outputs)
    return model
#--------------------------------------------------------------------------------------------------

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

def show_image(images, labels, classes):
    for i in range(min(5, len(images))):
        image = images[i].copy()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(''.join(classes[np.argmax(labels[char][i])] for char in range(num_chars)))
        plt.axis('off')
        plt.show()
        input("Press Enter to continue...")  

image_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/imgs"
annotations_dir = "/Users/hossein/Desktop/CV/PROJECTS/output/00/anns/xmls"


input_shape = (100, 450)  #  aspect ratio of 4.5 (100 * 4.5 = 450)

images, labels = load_data(image_dir, annotations_dir, input_shape)




#show_image(images, labels, classes)

split_index = int(0.8 * len(images))
train_images, val_images = images[:split_index], images[split_index:]
train_labels = [label_set[:split_index] for label_set in labels]
val_labels = [label_set[split_index:] for label_set in labels]

model = create_model(input_shape=(100, 450, 3), num_chars=num_chars, num_classes=num_classes)


model.compile(optimizer='adam', 
              loss=['categorical_crossentropy']*num_chars, 
              metrics=['accuracy']*num_chars)


history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=16)




val_loss, *val_accuracies = model.evaluate(val_images, val_labels)
average_accuracy = np.mean(val_accuracies)
print(f'Validation Accuracy: {average_accuracy * 100:.2f}%')



model.save("character_classification_model2.h5")


