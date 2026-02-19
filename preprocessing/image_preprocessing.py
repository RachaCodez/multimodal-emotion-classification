"""
Image preprocessing utilities.
- Detect face region with Haar cascades (fallback to full image)
- Resize and normalize to model input size
"""

import cv2
import numpy as np
from config import Config


def detect_face(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return image
    x, y, w, h = faces[0]
    face = image[y:y + h, x:x + w]
    return face


def preprocess_image(image_path: str):
    face = detect_face(image_path)
    if face is None:
        raise ValueError('Unable to read image file')
    face_resized = cv2.resize(face, Config.IMAGE_SIZE)
    face_normalized = face_resized.astype('float32') / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    return face_batch

