import sys
import os
import json
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO  # ✅ NEW Import for YOLO Gender Detection

# ✅ Fix for Torch + MediaPipe OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# App logic imports
from app.state import init_state, update_state
from models.body_analyzer import estimate_body_type
from models.color_detector import detect_skin_tone
from models.outfit_embedder import recommend_outfits
from models.tryon_renderer import render_outfit

# ✅ Load YOLOv8 Gender Detection Model
gender_model = YOLO("models/gender_detector.pt")  # Make sure this file exists

# Streamlit Config
st.set_page_config(page_title="Fashion Outfit Recommender", layout="centered")
st.title("👗 AI-Powered Fashion Outfit Recommender")

init_state()

# 📤 Upload Section
uploaded = st.file_uploader("📸 Upload your selfie or full-body photo", type=["jpg", "jpeg", "png"])

if uploaded:
    selfie_pil = Image.open(uploaded).convert("RGB")
    selfie_cv2 = cv2.cvtColor(np.array(selfie_pil), cv2.COLOR_RGB2BGR)

    # 📷 Display Uploaded Image
    st.markdown("### 📷 Your Uploaded Photo")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(selfie_pil.resize((300, 400)), caption="Selfie Preview", use_container_width=True)

    # ✅ 👤 Step 1: Detect Gender (Using YOLO)
    with st.spinner("Detecting Gender..."):
        results = gender_model(selfie_cv2)
        classes = results[0].boxes.cls.cpu().numpy()
        if len(classes) == 0:
            gender = "Unknown"
        else:
            gender = "MEN" if int(classes[0]) == 0 else "WOMEN"
    st.markdown(f"*Detected Gender:* {gender}")

    # 👕 Step 2: Detect Body Type
    body_type = estimate_body_type(selfie_cv2)
    st.markdown(f"*Estimated Body Type:* {body_type}")

    # 🎨 Step 3: Detect Skin Tone
    skin_tone = detect_skin_tone(selfie_cv2)
    st.markdown(f"*Detected Skin Tone:* {skin_tone}")

    # 👗 Step 4: Select Style
    style = st.selectbox("👗 Choose a style preference:", ["", "casual", "formal", "party"])

    if style:
        st.subheader("🧥 Outfit Suggestions")
        outfit_paths = recommend_outfits(gender, style)

        if not outfit_paths:
            st.warning("🚫 No outfits found for the selected options.")
        else:
            for i in range(0, len(outfit_paths), 3):
                row = st.columns(3)
                for j in range(3):
                    if i + j < len(outfit_paths):
                        path = outfit_paths[i + j]
                        with row[j]:
                            outfit_img = Image.open(path)
                            st.image(outfit_img.resize((200, 300)), caption=f"Option {i + j + 1}", use_container_width=True)

                            # Try-On Button
                            if st.button(f"👗 Try On {i + j + 1}", key=f"tryon_{i + j}"):
                                result = render_outfit(selfie_cv2, np.array(outfit_img))
                                if result is not None:
                                    st.image(result, caption="👗 Try-On Result", use_container_width=True)
                                    update_state("current_outfit", path)
                                else:
                                    st.error("❌ Failed to generate try-on image.")

                            # Like Button
                            if st.button(f"❤ Like {i + j + 1}", key=f"like_{i + j}"):
                                liked_file = "outputs/liked_outfits.json"
                                os.makedirs("outputs", exist_ok=True)
                                try:
                                    with open(liked_file, "r") as f:
                                        liked = json.load(f)
                                except FileNotFoundError:
                                    liked = []
                                liked.append({
                                    "path": path,
                                    "timestamp": str(datetime.now()),
                                    "gender": gender,
                                    "style": style,
                                    "body_type": body_type,
                                    "skin_tone": skin_tone
                                })
                                with open(liked_file, "w") as f:
                                    json.dump(liked, f, indent=2)
                                st.success(f"❤ Saved: {os.path.basename(path)}")
    else:
        st.warning("☝ Please choose a style to see recommended outfits.")
