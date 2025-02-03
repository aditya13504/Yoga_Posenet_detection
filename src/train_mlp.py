import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # <-- ADD THIS
import joblib

def load_keypoints(data_dir):
    X, y = [], []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for npy_file in os.listdir(class_dir):
            keypoints = np.load(os.path.join(class_dir, npy_file))
            X.append(keypoints)
            y.append(class_name)
    
    return np.array(X), np.array(y)

def calculate_confidence_thresholds(model, X_val, y_val_encoded):
    val_probs = model.predict_proba(X_val)
    thresholds = {}
    for class_idx in range(len(model.classes_)):
        mask = (y_val_encoded == class_idx)
        if np.sum(mask) > 0:
            class_probs = val_probs[mask, class_idx]
            thresholds[model.classes_[class_idx]] = np.percentile(class_probs, 25)
        else:
            thresholds[model.classes_[class_idx]] = 0.5
    return thresholds

if __name__ == "__main__":
    # Load data
    X, y = load_keypoints("keypoints\TRAIN")  #enter TRAIN path
    X_test, y_test = load_keypoints("keypoints\TEST")   #enter TEST path
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Split validation set
    X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train, y_train_encoded)  # Use encoded labels
    
    # Calculate thresholds
    confidence_thresholds = calculate_confidence_thresholds(mlp, X_val, y_val_encoded)
    
    # Evaluate
    y_pred_encoded = mlp.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlp, "models/pose_classifier.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")  # Save encoder
    np.save("models/confidence_thresholds.npy", confidence_thresholds)