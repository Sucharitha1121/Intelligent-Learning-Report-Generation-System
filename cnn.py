from google.colab import drive
drive.mount('/content/drive')

!pip install pillow openpyxl tensorflow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from openpyxl import Workbook
import cv2
import os

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
input_shape = (28, 28, 1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=30, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)

def preprocess_image(image):
    img = Image.fromarray(image).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype('float32') / 255  # Normalize
    img_array = 1 - img_array  # Invert colors if necessary
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to match model input
    return img_array

def is_blank_region(region, intensity_threshold=200, variance_threshold=50):
    """
    Check if the region is blank based on mean intensity and variance.

    Parameters:
    - region: The image region to check.
    - intensity_threshold: The mean intensity threshold to consider the region as blank.
    - variance_threshold: The variance threshold to consider the region as blank.

    Returns:
    - bool: True if the region is blank, False otherwise.
    """
    mean_intensity = np.mean(region)
    variance = np.var(region)

    return mean_intensity > intensity_threshold and variance < variance_threshold

def predict_and_write_to_excel(image_paths, coordinates, headings, excel_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Handwritten Digit Predictions"

    # Write header
    ws.append(["S.No"] + headings)

    for idx, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(image_path)
        predictions = []

        for coord in coordinates:
            x1, y1, x2, y2 = coord
            digit_img = img[y1:y2, x1:x2]

            # Check if the region is mostly white and has low variance (indicating it's blank)
            if is_blank_region(digit_img):
                predictions.append("")
            else:
                input_image = preprocess_image(digit_img)
                pred = model.predict(input_image)
                predicted_digit = pred.argmax()
                predictions.append(predicted_digit)

                # Display the image and prediction
                plt.imshow(digit_img, cmap='Greys')
                plt.title(f"Predicted Digit: {predicted_digit}")
                plt.show()

        ws.append([idx] + predictions)

    wb.save(excel_path)
    print(f"Predictions written to {excel_path}")

# List of image paths
image_paths = [
    "/content/drive/My Drive/epics/marksheets/1.bmp",
    "/content/drive/My Drive/epics/marksheets/3.bmp",
    "/content/drive/My Drive/epics/marksheets/4.bmp",
    "/content/drive/My Drive/epics/marksheets/7.bmp",
    "/content/drive/My Drive/epics/marksheets/8.bmp",
    "/content/drive/My Drive/epics/marksheets/10.bmp",
    "/content/drive/My Drive/epics/marksheets/11.bmp",
    "/content/drive/My Drive/epics/marksheets/12.bmp",
    "/content/drive/My Drive/epics/marksheets/13.bmp",
    "/content/drive/My Drive/epics/marksheets/14.bmp",
    "/content/drive/My Drive/epics/marksheets/15.bmp"
]

# Define the coordinates
coordinates = [
    # Initial coordinates for a, b, c, d, e, f, g, h, i, j
    (325, 227, 359, 260),  # Q.No a
    (365, 228, 395, 259),  # Q.No b
    (401, 228, 429, 259),  # Q.No c
    (436, 227, 470, 259),  # Q.No d
    (473, 229, 506, 258),  # Q.No e
    (508, 228, 541, 259),  # Q.No f
    (544, 228, 577, 259),  # Q.No g
    (581, 228, 615, 259),  # Q.No h
    (619, 230, 654, 259),  # Q.No i
    (657, 229, 688, 259),  # Q.No j
    # New coordinates for 2a, 2b, 3a, 3b, etc.
    (77, 315, 111, 345), (118, 319, 147, 345),  # 2a, 2b
    (325, 318, 358, 344), (362, 317, 396, 344),  # 3a, 3b
    (77, 351, 110, 376), (119, 351, 148, 379),  # 4a, 4b
    (329, 350, 358, 378), (363, 350, 396, 378),  # 5a, 5b
    (78, 383, 113, 413), (117, 383, 148, 413),  # 6a, 6b
    (325, 383, 359, 411), (362, 383, 398, 410),  # 7a, 7b
    (78, 416, 114, 444), (117, 416, 149, 444),  # 8a, 8b
    (325, 415, 359, 444), (362, 415, 398, 444)  # 9a, 9b
]

# Define the headings
headings = [
    '1a', '1b', '1c', '1d', '1e', '1f', '1g', '1h', '1i', '1j',
    '2a', '2b', '3a', '3b', '4a', '4b', '5a', '5b', '6a', '6b', '7a', '7b', '8a', '8b', '9a', '9b'
]

# Path to save the Excel file
excel_path = '/content/drive/My Drive/epics/marksheets/digit_predictions.xlsx'

# Predict digits and write to Excel
predict_and_write_to_excel(image_paths, coordinates, headings, excel_path)