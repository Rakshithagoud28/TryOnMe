import cv2
import numpy as np

def render_outfit(selfie_cv2, outfit_img_np):
    """Overlays outfit image on selfie — Basic Try-On (placeholder)."""
    try:
        outfit_resized = cv2.resize(outfit_img_np, (selfie_cv2.shape[1], selfie_cv2.shape[0]))
        blended = cv2.addWeighted(selfie_cv2, 0.6, outfit_resized, 0.4, 0)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error in try-on rendering: {e}")
        return None
