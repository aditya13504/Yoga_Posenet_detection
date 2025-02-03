import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_keypoints(image_path):
    """Extracts PoseNet keypoints from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract (x, y) coordinates of 33 PoseNet landmarks
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y])
    
    return np.array(keypoints)

def save_average_poses(keypoints_dir, output_dir):
    """Calculates and saves average keypoints for each pose."""
    os.makedirs(output_dir, exist_ok=True)
    
    for pose_name in os.listdir(keypoints_dir):
        pose_dir = os.path.join(keypoints_dir, pose_name)
        if not os.path.isdir(pose_dir):
            continue
        
        all_keypoints = []
        for npy_file in os.listdir(pose_dir):
            keypoints = np.load(os.path.join(pose_dir, npy_file))
            all_keypoints.append(keypoints)
        
        if len(all_keypoints) > 0:
            avg_keypoints = np.mean(all_keypoints, axis=0)
            np.save(os.path.join(output_dir, f"{pose_name}.npy"), avg_keypoints)

def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            keypoints = extract_keypoints(img_path)
            
            if keypoints is not None:
                output_path = os.path.join(output_dir, class_name, f"{os.path.splitext(img_file)[0]}.npy")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, keypoints)

if __name__ == "__main__":
    # Process training and test data
    process_dataset("dataset\TRAIN", "keypoints\TRAIN")  #enter TRAIN dataset and keypoints path
    process_dataset("dataset\TEST", "keypoints\TEST")   #enter TEST dataset and keypoints path
    
    # Create average poses
    save_average_poses("keypoints/TRAIN", "average_poses")   #enter TRAIN keypoints path