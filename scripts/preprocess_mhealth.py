import os
import numpy as np
import pandas as pd
import json
from scipy.signal import resample

# ---------------------------
# Config
# ---------------------------
TARGET_HZ = 20
WINDOW_SEC = 10
WINDOW_SIZE = TARGET_HZ * WINDOW_SEC  # 200

# ---------------------------
# Load mHealth
# ---------------------------
def load_mhealth(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None)
    
    # Last column = label
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    return data, labels

# ---------------------------
# Resample
# ---------------------------
def resample_signal(data, original_hz=50, target_hz=20):
    num_samples = int(data.shape[0] * target_hz / original_hz)
    return resample(data, num_samples)

# ---------------------------
# Windowing
# ---------------------------
def create_windows(data, labels):
    X, y = [], []
    
    for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
        window = data[i:i+WINDOW_SIZE]
        label = labels[i:i+WINDOW_SIZE]
        
        # Majority label
        label = np.bincount(label.astype(int)).argmax()
        
        # Skip null class (0)
        if label == 0:
            continue
        
        X.append(window)
        y.append(label)
    
    return np.array(X), np.array(y)

# ---------------------------
# Main
# ---------------------------
def process_mhealth(data_dir):
    print("Processing mHealth (Bonus HAR) ...")
    
    file_path = os.path.join(data_dir, "data/raw/MHEALTHDATASET/mHealth_subject1.log")
    
    data, labels = load_mhealth(file_path)
    
    # Resample
    data_resampled = resample_signal(data, original_hz=50, target_hz=20)
    
    # Align labels length
    labels_resampled = labels[:len(data_resampled)]
    
    # Windowing
    X, y = create_windows(data_resampled, labels_resampled)
    
    print("mHealth shape:", X.shape)
    
    # Metadata
    metadata = {
        "dataset": "mHealth",
        "sampling_rate": 20,
        "window_size": WINDOW_SIZE,
        "null_class_handling": "removed (label 0 excluded)",
        "channels": X.shape[-1] if len(X) > 0 else 0
    }
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("manifest", exist_ok=True)
    
    np.save("data/processed/X_mhealth.npy", X)
    np.save("data/processed/y_mhealth.npy", y)
    
    with open("manifest/mhealth_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("mHealth pipeline completed!")
    
    return X, y, metadata

# --------------------------
if __name__ == "__main__":

    process_mhealth(os.getcwd())
