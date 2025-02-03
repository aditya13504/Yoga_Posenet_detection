# Yoga Pose Detection with PoseNet and MLP 🧘️✨

Real-time yoga pose detection and feedback system using **MediaPipe PoseNet** and an **MLP classifier**. Detects 5 yoga poses and provides visual guidance to correct posture.

## Features 🚀
- Real-time pose detection using webcam
- Compares user’s pose with ideal posture using keypoints
- Visual feedback with colored landmarks (✅ Correct, ❌ Adjustments needed)
- Confidence-based predictions with thresholds
- Detailed limb difference analysis

# Project Structure 📂
yoga-pose-detection/
├── dataset/               
├── keypoints/             
├── average_poses/         
├── models/                
├── src/
│   ├── extract_keypoints.py
│   ├── train_mlp.py
│   └── detect_pose.py
├── docs/                  
├── requirements.txt
└── README.md

## Installation ⚙️

### 1. Clone the Repository
- open cmd/powershell/vs code
- git clone https://github.com/your-username/yoga-pose-detection.git
- cd yoga-pose-detection
- pip install -r requirements.txt

## Dataset
Download a yoga pose dataset (https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset).

Organize it into dataset/TRAIN and dataset/TEST folders with subfolders for each pose:

dataset/
├── TRAIN/
│   ├── downdog/
│   ├── goddess/
│   └── ...
└── TEST/
    ├── downdog/
    └── ...

# Usage 🖥️
- run:
- python src/extract_keypoints.py
- python src/train_mlp.py
- python src/detect_pose.py

## Working 🔍
PoseNet extracts 33 body keypoints from input images/video.
Keypoints are saved as .npy files and averaged for ideal poses.
MLP Classifier predicts the pose using keypoint coordinates.
Feedback System compares user keypoints with ideal poses and highlights discrepancies.
