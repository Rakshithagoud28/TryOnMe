import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import random
import shutil

# Load YOLOv8 Gender Detection Model
model = YOLO("models/gender_detector.pt")

# Outfit folders
OUTFIT_FOLDER = "data"
MEN_FOLDER = os.path.join(OUTFIT_FOLDER, "MEN")
WOMEN_FOLDER = os.path.join(OUTFIT_FOLDER, "WOMEN")

def detect_gender(image_path):
    results = model(image_path)
    classes = results[0].boxes.cls.cpu().numpy()
    if len(classes) == 0:
        return "Unknown"
    class_id = int(classes[0])
    return "MEN" if class_id == 0 else "WOMEN"

def suggest_outfits(gender, num_outfits=3):
    folder = MEN_FOLDER if gender == "MEN" else WOMEN_FOLDER
    outfit_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not outfit_files:
        return []
    return random.sample(outfit_files, min(num_outfits, len(outfit_files)))

st.title("👕 TryOnMe - Gender Detection & Outfit Recommender")

uploaded_file = st.file_uploader("Upload Your Selfie or Full-length Image 👇", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("Analyzing Gender..."):
        image = Image.open(uploaded_file)
        temp_path = "temp_uploaded_image.jpg"
        image.save(temp_path)

        gender = detect_gender(temp_path)
        st.image(image, caption=f"Detected Gender: {gender}", use_column_width=True)

        if gender in ["MEN", "WOMEN"]:
            st.subheader(f"Suggested Outfits for {gender}")
            outfits = suggest_outfits(gender)
            for outfit in outfits:
                st.image(os.path.join(MEN_FOLDER if gender == "MEN" else WOMEN_FOLDER, outfit), caption=outfit, use_column_width=True)
        else:
            st.warning("No Gender Detected. Please upload a clearer image.")

        os.remove(temp_path)
