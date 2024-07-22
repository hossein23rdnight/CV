

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError

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
        Dense(8)
    ])

    return model


images, labels = read_and_normalize_data(image_folder, label_folder)

# rotated_images, rotated_labels = rotate_images_and_labels(images, labels)
# blurred_images, blurred_labels = blur_images_and_labels(images, labels, max_kernel_size=3)
# contrast_images,contrast_labels = adjust_contrast_images_and_labels(images, labels)
# transformed_images,transformed_labels = perspective_transform_images_and_labels(images, labels)
# crop_images,crop_labels = crop_images_and_labels(images, labels)
# shift_images,shift_labels = shift_images_and_labels(images, labels)


# all_images = np.concatenate([images, rotated_images, blurred_images, contrast_images, transformed_images, crop_images, shift_images], axis=0)
# all_labels = np.concatenate([labels, rotated_labels, blurred_labels, contrast_labels, transformed_labels, crop_labels, shift_labels], axis=0)


all_images=images
all_labels=labels

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

