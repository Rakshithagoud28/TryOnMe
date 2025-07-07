import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_body_landmarks(image):
    """Detects body keypoints using MediaPipe Pose."""
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        else:
            return None

def render_outfit(selfie_cv2, outfit_img_np):
    """Fits outfit from neck to waist, matching user’s upper body region with enhanced blending."""
    try:
        landmarks = detect_body_landmarks(selfie_cv2)
        if landmarks is None:
            print("❌ No pose detected, falling back to whole-body blend.")
            return fallback_blend(selfie_cv2, outfit_img_np)

        h, w, _ = selfie_cv2.shape

        # Key Points
        shoulder_left = landmarks[11]
        shoulder_right = landmarks[12]
        hip_left = landmarks[23]
        hip_right = landmarks[24]
        
        # Calculate neck to waist box
        x_min = int(min(shoulder_left.x, shoulder_right.x) * w)
        x_max = int(max(shoulder_left.x, shoulder_right.x) * w)
        y_neck = int(min(shoulder_left.y, shoulder_right.y) * h)
        y_waist = int((hip_left.y + hip_right.y) / 2 * h)

        # Ensure valid coordinates
        if y_neck >= y_waist or x_min >= x_max:
            print("❌ Invalid landmark positions detected.")
            return fallback_blend(selfie_cv2, outfit_img_np)

        # Resize outfit to fit between neck & waist width
        outfit_resized = cv2.resize(
            outfit_img_np, (x_max - x_min, y_waist - y_neck)
        )

        # Brighten outfit for visibility
        bright_outfit = cv2.convertScaleAbs(outfit_resized, alpha=1.3, beta=30)

        # Blend more towards outfit (stronger outfit visibility)
        result = selfie_cv2.copy()
        roi = result[y_neck:y_waist, x_min:x_max]
        blended = cv2.addWeighted(roi, 0.3, bright_outfit, 0.7, 0)
        result[y_neck:y_waist, x_min:x_max] = blended

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Error in try-on rendering: {e}")
        return None

def fallback_blend(selfie_cv2, outfit_img_np):
    """Fallback to whole-image blending (less realistic)."""
    outfit_resized = cv2.resize(outfit_img_np, (selfie_cv2.shape[1], selfie_cv2.shape[0]))
    bright_outfit = cv2.convertScaleAbs(outfit_resized, alpha=1.3, beta=30)
    blended = cv2.addWeighted(selfie_cv2, 0.3, bright_outfit, 0.7, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
