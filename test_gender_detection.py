import cv2
from models.gender_classifier import detect_gender  # adjust path if needed

# Replace with the path to a real image (selfie)
img = cv2.imread("test_images/mypic1.jpg")


if img is None:
    print("❌ Failed to load image. Check the path.")
else:
    gender = detect_gender(img)
    print("✅ Detected gender:", gender)