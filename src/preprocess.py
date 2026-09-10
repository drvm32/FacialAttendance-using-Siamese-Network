import cv2
import numpy as np
import os

def detect_and_align_face(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    return face / 255.0

def load_reference_images(directory):
    ref_images = {}
    for filename in os.listdir(directory):
        person_id = filename.split('.')[0]
        img = detect_and_align_face(os.path.join(directory, filename))
        if img is not None:
            ref_images[person_id] = img
    return ref_images

if __name__ == "__main__":
    refs = load_reference_images('../data/reference_images/')
    print(f"Loaded {len(refs)} reference images: {list(refs.keys())}")