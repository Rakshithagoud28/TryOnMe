# 👗 TryOnMe-Fashion — AI-Powered Fashion Outfit Recommender
TryOnMe-Fashion is an AI fashion outfit recommender app using YOLOv8, Streamlit, and PyTorch. It detects gender from photos to suggest personalized outfits with virtual try-on features.


---

## 🚀 Features:
- ✅ Gender Detection (YOLOv8 custom model)
- ✅ Body Type Estimation
- ✅ Skin Tone Detection
- ✅ Style-Based Outfit Recommendations (casual, formal, party)
- ✅ Virtual Try-On (Outfit overlay on uploaded photo)
- ✅ Save Favorite Outfits (Like Button)

---

## 📦 Folder Structure:
├── app/ # Streamlit app state logic
├── data/ # Dataset & Outfit Images
├── models/ # Models for gender, body, color, try-on, embedding
├── outputs/ # Saved liked outfits (JSON)
├── runs/ # YOLO training and predictions
├── scripts/ # Helper scripts (dataset processing, embeddings, etc.)
├── venv/ # Python Virtual Environment (ignored in .gitignore)
├── yolov8n.pt # YOLOv8 base model
├── README.md # Project documentation
├── requirements.txt # Project dependencies
├── main.py # Streamlit app entry point
└── .gitignore # Ignored files

## 🧑‍💻 Tech Stack:
- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **Torch (PyTorch)**
- **Streamlit**
- **OpenCV**
- **Pillow (PIL)**
- **MediaPipe (for body analysis)**