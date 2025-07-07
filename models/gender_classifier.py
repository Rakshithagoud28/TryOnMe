import cv2

def detect_gender(image_cv2):
    """Detects gender from a selfie using face detection and basic heuristics."""
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "Unknown"

    # Dummy heuristic (for demo) — You can replace with a better model later
    (x, y, w, h) = faces[0]
    face_area = w * h
    image_area = image_cv2.shape[0] * image_cv2.shape[1]

    ratio = face_area / image_area

    if ratio > 0.05:
        return "WOMEN"
    else:
        return "MEN"
