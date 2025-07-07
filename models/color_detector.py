import cv2
import numpy as np

def detect_skin_tone(image_cv2):
    """Detects dominant skin tone in selfie (based on face region)."""
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "Unknown"

    (x, y, w, h) = faces[0]
    face_region = image_cv2[y:y+h, x:x+w]

    pixels = face_region.reshape((-1, 3))
    pixels = np.float32(pixels)

    n_colors = 1  # Detect dominant skin tone
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[0].astype(int)

    # Simple color label (can be improved later)
    if dominant_color[2] > 150:
        return "Fair"
    elif dominant_color[2] > 100:
        return "Medium"
    else:
        return "Dark"
