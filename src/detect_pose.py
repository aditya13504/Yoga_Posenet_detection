import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load model and artifacts
mlp = joblib.load("models/pose_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
confidence_thresholds = np.load("models/confidence_thresholds.npy", allow_pickle=True).item()

# Load average poses
average_poses = {}
for pose_file in os.listdir("average_poses"):
    pose_name = os.path.splitext(pose_file)[0]
    average_poses[pose_name] = np.load(os.path.join("average_poses", pose_file))

def calculate_limb_differences(user_kp, ideal_kp):
    """Calculates differences in limb lengths between poses"""
    limbs = [
        (11, 13), (13, 15),   # Right arm
        (12, 14), (14, 16),   # Left arm
        (23, 25), (25, 27),   # Right leg
        (24, 26), (26, 28)    # Left leg
    ]
    
    differences = []
    for (i, j) in limbs:
        user_dist = np.linalg.norm(user_kp[i] - user_kp[j])
        ideal_dist = np.linalg.norm(ideal_kp[i] - ideal_kp[j])
        differences.append(abs(user_dist - ideal_dist))
    
    return np.mean(differences)

def create_ideal_landmarks(ideal_kp):
    """Creates a custom object to represent ideal pose landmarks."""
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class PoseLandmark:
        def __init__(self, keypoints):
            self.landmark = [Landmark(x, y) for x, y in keypoints]

    return PoseLandmark(ideal_kp)

def draw_comparison(frame, user_landmarks, ideal_kp):
    h, w, _ = frame.shape
    
    # Create ideal landmarks
    ideal_landmarks = create_ideal_landmarks(ideal_kp)
    
    # Draw user pose (green)
    mp_drawing.draw_landmarks(
        frame, user_landmarks, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )
    
    # Draw ideal pose (blue)
    for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        cx = int(ideal_landmarks.landmark[idx].x * w)
        cy = int(ideal_landmarks.landmark[idx].y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    
    # Draw connections between mismatched joints
    for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        user_x = int(user_landmarks.landmark[idx].x * w)
        user_y = int(user_landmarks.landmark[idx].y * h)
        ideal_x = int(ideal_landmarks.landmark[idx].x * w)
        ideal_y = int(ideal_landmarks.landmark[idx].y * h)
        
        if abs(user_x - ideal_x) > 50 or abs(user_y - ideal_y) > 50:
            cv2.line(frame, (user_x, user_y), (ideal_x, ideal_y), (0, 0, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract keypoints
            user_kp = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark]).flatten()
            
            # Predict pose
            try:
                pose_encoded = mlp.predict([user_kp])[0]
                confidence = np.max(mlp.predict_proba([user_kp]))
                pose_name = label_encoder.inverse_transform([pose_encoded])[0]
            except:
                pose_name = "unknown"
                confidence = 0.0
            
            # Get ideal pose
            ideal_kp = average_poses.get(pose_name, None)
            
            if ideal_kp is not None and confidence > confidence_thresholds.get(pose_name, 0.5):
                # Calculate differences
                user_kp_2d = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
                ideal_kp_2d = np.reshape(ideal_kp, (33, 2))
                
                limb_diff = calculate_limb_differences(user_kp_2d, ideal_kp_2d)
                
                # Display feedback
                if limb_diff > 0.15:
                    draw_comparison(frame, results.pose_landmarks, ideal_kp_2d)
                    cv2.putText(frame, f"Adjust your {pose_name}!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Good {pose_name}! ({confidence*100:.1f}%)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Pose not recognized", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Yoga Pose Correction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Yoga Pose Correction', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()