import cv2
import mediapipe as mp

def estimate_body_type(image_cv2):
    """Estimates body type using MediaPipe Pose landmarks."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return "Unknown"

    # Dummy rule-based estimation (replace with ML later if needed)
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    hip_width = abs(left_hip.x - right_hip.x)

    if shoulder_width > hip_width:
        return "Inverted Triangle"
    elif hip_width > shoulder_width:
        return "Pear"
    else:
        return "Rectangle"
