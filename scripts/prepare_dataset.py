import os
import pandas as pd
from PIL import Image

DATA_DIR = "data/images"
OUTPUT_CSV = "data/metadata.csv"

metadata = []

for gender in os.listdir(DATA_DIR):
    gender_path = os.path.join(DATA_DIR, gender)
    if not os.path.isdir(gender_path):
        continue

    for style in os.listdir(gender_path):
        style_path = os.path.join(gender_path, style)
        if not os.path.isdir(style_path):
            continue

        for filename in os.listdir(style_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(style_path, filename)

                # ✅ Resize image to 224x224 (standard)
                img = Image.open(file_path).convert("RGB")
                img = img.resize((224, 224))
                img.save(file_path)  # Overwrite resized image

                # ✅ Add to metadata
                relative_path = os.path.join(gender, style, filename)
                metadata.append({
                    "filename": os.path.join(DATA_DIR, relative_path).replace("\\", "/"),
                    "gender": gender,
                    "style": style
                })

# ✅ Save metadata CSV
df = pd.DataFrame(metadata)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Dataset prepared and metadata saved to {OUTPUT_CSV}")
