# 👗 TryOnMe - AI-Powered Fashion Outfit Recommender

TryOnMe is an AI-powered web app that recommends personalized outfits based on your selfie.  
It analyzes **gender**, **body type**, **skin tone**, and your **style preferences** to suggest suitable outfits, allowing you to preview, try on, and save favorites.

---

## 🚀 Features
- 📸 Upload your selfie or full-body photo
- 👤 Detects gender automatically
- 🔍 Analyzes body type using pose estimation
- 🎨 Detects skin tone from your photo
- ✨ Outfit recommendations based on your style preference (casual, formal, party)
- 🧥 Virtual Try-On: Overlay selected outfits on your image
- ❤️ Like and save favorite outfits

---

## 🧰 Tech Stack
- **Languages:** Python
- **AI & CV:** OpenCV, MediaPipe, CLIP, ResNet, FAISS
- **Frameworks:** Streamlit
- **Others:** Pillow, NumPy, Torch, Scikit-learn, DeepFashion Dataset

---

## 📂 Project Structure

fashion-outfit-recommender/
│
├── app/ # Streamlit app logic
│ ├── main.py # Main app flow
│ └── state.py # App state management
│
├── data/ # Dataset & embeddings
│ ├── images/ # Outfit images
│ └── metadata.csv # Outfit metadata
│
├── models/ # Core detection and recommendation models
│
├── outputs/ # Saved user-liked outfits
│
├── scripts/ # Data preparation scripts
│
├── requirements.txt # Project dependencies
└── README.md # Project documentation

 

