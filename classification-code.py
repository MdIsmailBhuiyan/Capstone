#python -m venv myenv  
#myenv\Scripts\activate
# .\venv\Scripts\activate

#python classification-code.py

#pip install pydicom numpy matplotlib opencv-python tensorflow scikit-learn Pillow
#or,
#pip install -r requirement.txt


import os
import pydicom
import numpy as np
import cv2

from PIL import Image
Fimage = []
def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / np.max(image_resized)
    return image_normalized
def load_dicom_images_from_folder(base_folder):
    images = []
    labels = []
    for patient_folder in os.listdir(base_folder):
        patient_path = os.path.join(base_folder, patient_folder)
        if os.path.isdir(patient_path):
            for study_folder in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_folder)
                if os.path.isdir(study_path):
                    for series_folder in os.listdir(study_path):
                        series_path = os.path.join(study_path, series_folder)
                        if os.path.isdir(series_path):
                            for dicom_file in os.listdir(series_path):
                                if dicom_file.endswith('.dcm'):
                                    dicom_path = os.path.join(series_path, dicom_file) # File link
                                    image = pydicom.dcmread(dicom_path).pixel_array


                                    Fimage.append(pydicom.dcmread(dicom_path))


                                    #JPG Converter
                                    
                                    # Check if the pixel array is in the expected format (for grayscale)
                                    if image.ndim == 2:  # Grayscale image
                                        image1 = Image.fromarray(image).convert('L')
                                    elif image.ndim == 3:  # RGB image
                                        image1 = Image.fromarray(image, mode='RGB')
                                    else:
                                        raise ValueError("Unsupported image format")
                                    
                                    # Convert to a mode supported by JPEG
                                    if image1.mode == 'I':
                                        image1 = image.convert('L')  # Convert to grayscale
                                    # Construct output file path
                                    output_folder = os.path.join(series_path , 'JPG Folder')
                                    os.makedirs(output_folder, exist_ok=True)  # Create folder
                                    jpg_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(dicom_file))[0] + '.jpg')
                                    # Save the image as JPG
                                    image1.save(jpg_file_path, 'JPEG', quality=100)
                                    #print(f'Saved: {jpg_file_path}')

                                    image = preprocess_image(image)
                                    images.append(image)
                                    labels.append(patient_folder)  # Label based on patient folder

    return np.array(images), np.array(labels)

from sklearn.model_selection import train_test_split

base_folder = 'G:\\Datasets\\test\\TCGA-PRAD'  # Replace with your actual path
images, labels = load_dicom_images_from_folder(base_folder)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



print("DICOM File Information:")
for elem in Fimage[0].iterall():
    print(f"{elem.tag} ({elem.name}): {elem.value}")


"""
import matplotlib.pyplot as plt

# Display the pixel array as an image
plt.imshow(X_train[0], cmap=plt.cm.gray)
plt.show()



import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.evaluate(X_test, y_test)

"""