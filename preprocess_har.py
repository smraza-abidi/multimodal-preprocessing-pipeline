import numpy as np
import pandas as pd
import os

# *************************************
# Config (temporary - later from yaml)
# *************************************
# TARGET_SR = 20
# WINDOW_PRETRAIN = 10   # seconds
# WINDOW_SUP = 5         # seconds
# OVERLAP = 0.5

# ***************************************
# Utility: Resample (Sampling Frequency)
# ***************************************
def resample_signal(data, labels, subjects, original_sr, target_sr):
    factor = int(original_sr / target_sr)

    # Downsample
    data_resampled = data[::factor]
    labels_resampled = labels[::factor]
    subjects_resampled = subjects[::factor]

    return data_resampled, labels_resampled, subjects_resampled

# **********************************
# Utility: Windowing (Segmentation)
# **********************************
def create_windows(data, labels, subjects, window_size=200, step_size=100):
    
    X = []
    y = []
    s = []

    for i in range(0, len(data) - window_size, step_size):

        window_data = data[i:i+window_size]
        window_labels = labels[i:i+window_size]
        window_subjects = subjects[i:i+window_size]

        # Majority label
        label = np.bincount(window_labels).argmax()

        # Majority subject
        subject = np.bincount(window_subjects.astype(int)).argmax()

        X.append(window_data)
        y.append(label)
        s.append(subject)

    return np.array(X), np.array(y), np.array(s)

# ***********************************************
# Loader Function: PAMAP2 Dataset (6 - channels)
# ***********************************************
def load_pamap2(file_path):
    # Load dataset
    df = pd.read_csv(file_path, sep="\s+", header=None)

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Wrist IMU Hand (3D-acceleration data and 3D-gyroscope data) As per documentation (readme)
    acc_phone = df.iloc[:, [4,5,6]].values
    gyro_phone = df.iloc[:, [10,11,12]].values

    # Label column
    labels = df.iloc[:,1].values

    data = np.hstack((acc_phone, gyro_phone)) # (N, 6)    
   
    return data, labels

# **************************************
# Loader Function: WISDM (6 - channels)
# **************************************
def load_wisdm(accel_path, gyro_path):
    # Load datasets (accel and gyro) from phone
    acc_df = pd.read_csv(accel_path, sep=",", header=None)
    gyro_df = pd.read_csv(gyro_path, sep=",", header=None)

    # Remove semicolon from last column
    acc_df[5] = acc_df[5].astype(str).str.replace(";", "", regex=False)
    gyro_df[5] = gyro_df[5].astype(str).str.replace(";", "", regex=False)

    # Assign column names (as per WISDM format based on dataset documentation)
    acc_df.columns = ["user","activity","timestamp","acc_x","acc_y","acc_z"]
    gyro_df.columns = ["user","activity","timestamp","gyro_x","gyro_y","gyro_z"]

    # Align lengths (IMPORTANT)
    min_len = min(len(acc_df), len(gyro_df))
    acc_df = acc_df.iloc[:min_len]
    gyro_df = gyro_df.iloc[:min_len]

    # Combine signals
    acc = acc_df[["acc_x", "acc_y", "acc_z"]].values
    gyro = gyro_df[["gyro_x", "gyro_y", "gyro_z"]].values   

    data_wisdm = np.hstack((acc, gyro)) # (N, 6)

    # Label take from accel or gyro because activity column is same in both dataframes.
    # Label based on activity code (Range: A-S)
    label_wisdm_all = acc_df["activity"].values

    subjects = acc_df["user"].values  # Preservance of subject-ids 
     
    return data_wisdm, label_wisdm_all, subjects

# ***********************************************
# Label Harmanization: PAMAP2 and WISDM Datasets
# ***********************************************
# PAMAP2 label mapping
def map_pamap_labels(labels):

    mapped = []

    for l in labels:
        if l == 0:
            mapped.append(-1) # Remove activityID=0 as per PAMAP2 document
        elif l == 4:
            mapped.append(0)  # walking
        elif l == 5:
            mapped.append(1)  # running
        elif l == 2:
            mapped.append(2)  # sitting
        elif l == 3:
            mapped.append(3)  # standing
        else:
            mapped.append(-1)

    return np.array(mapped)

# WISDM label mapping
def map_wisdm_labels(labels):

    mapped = []

    for l in labels:
        
        if "A" in l:          
            mapped.append(0)  # Walking
        elif "B" in l:
            mapped.append(1)  # Jogging
        elif "D" in l:
            mapped.append(2)  # Sitting
        elif "E" in l:
            mapped.append(3)  # Standing
        else:
            mapped.append(-1)  # Unknown

    return np.array(mapped)

# *************************
# Main Function Processing
# *************************
def process_har():
    print("Starting HAR preprocessing ...")
    print("\nStarting PAMAP Processing ...")

    # ------------------------------------------------
    # STEP-1: Processing of PAMAP2 and WISDM Datasets
    # ------------------------------------------------
    # PAMAP path
    pamap_path = "data/raw/PAMAP2/pamap2/Protocol"
    
    all_data = []
    all_labels = []
    all_subjects = []
        
    for file in os.listdir(pamap_path):
        if file.endswith(".dat"):
            subject_id = int(file.replace("subject", "").replace(".dat", ""))

            data, labels = load_pamap2(os.path.join(pamap_path, file))

            all_data.append(data)
            all_labels.append(labels)
            all_subjects.extend([subject_id] * len(data))

            print(f"Loaded {subject_id}: {data.shape}")

    data_pamap = np.vstack(all_data)
    labels_pamap = np.concatenate(all_labels)
    subjects_pamap = np.array(all_subjects)    # Preservance of subject-ids

    print("Loaded datasets!")
    print('Combined Shape:', data_pamap.shape)

    # Start WISDM dataset
    print('\nStarting WISDM Processing ... ')

    # WISDM path
    wisdm_dir = "data/raw/WISDM/wisdm/accel_gyro"
    
    wisdm_data_list = []
    wisdm_labels_list = []
    wisdm_subjects = []

    for file in os.listdir(wisdm_dir):

        # Only process accel files
        if "_accel_" in file:
        
            # Extract subject ID (e.g., 1600)
            subject_id = file.split("_")[1]

            accel_path = os.path.join(wisdm_dir, file)

            # Construct matching gyro filename
            gyro_file = f"data_{subject_id}_gyro_phone.txt"
            gyro_path = os.path.join(wisdm_dir, gyro_file)

            if not os.path.exists(gyro_path):
                print(f"[WARNING:] Missing gyro for {subject_id}")
                continue

            # Load data
            data, labels, subjects = load_wisdm(accel_path, gyro_path)

            wisdm_data_list.append(data)
            wisdm_labels_list.append(labels)
            wisdm_subjects.extend(subjects)

            print(f"Loaded subject {subject_id}: {data.shape}")

    # Safety check
    if len(wisdm_data_list) == 0:
        raise ValueError("No WISDM files loaded. Check filenames.")

    wisdm_data = np.vstack(wisdm_data_list)
    wisdm_labels = np.concatenate(wisdm_labels_list)
    wisdm_subjects = np.array(wisdm_subjects)

    print("\nLoaded Datasets!")
    print(f"WISDM combined shape: {wisdm_data.shape}")

    # --------------------------------------
    # STEP-2: Resampling (100 Hz -> 20 Hz )
    # --------------------------------------
    # Applying (Resample PAMAP2 dataset)
    print("\nResampling PAMAP2 to 20 Hz ...")

    data_pamap, labels_pamap, subjects_pamap = resample_signal(
        data_pamap,
        labels_pamap,
        subjects_pamap,
        original_sr=100, # Sampling frequency 100 Hz
        target_sr=20     # Resampling to 20 Hz
    )
    print(f"PAMAP2 Resampled Shape: {data_pamap.shape}")

    print(len(data_pamap), len(labels_pamap), len(subjects_pamap))

    # Resample WISDM dataset (No Change)
    print("\nWISDM assumed already at 20 Hz")
    
    # ----------------------------
    # STEP-3: Label Mapping 
    # ----------------------------
    # Apply label mapping
    print("\nMapping labels ...")

    pamap_labels = map_pamap_labels(labels_pamap)
    wisdm_labels = map_wisdm_labels(wisdm_labels)

    # Remove unkown labels
    # PAMAP2
    mask = pamap_labels != -1
    pamap_data = data_pamap[mask]
    pamap_labels = pamap_labels[mask]
    pamap_subjects = subjects_pamap[mask]

    # WISDM
    mask = wisdm_labels != -1
    wisdm_data = wisdm_data[mask]
    wisdm_labels = wisdm_labels[mask]
    wisdm_subjects = wisdm_subjects[mask]

    # Sanity check
    print("Unique PAMAP labels:", np.unique(pamap_labels))
    print("Unique WISDM labels:", np.unique(wisdm_labels))

    # ------------------------
    # STEP-4: Apply Windowing 
    # ------------------------
    print("\nCreating Windows ...")

    # PAMAP2 dataset
    X_pamap, y_pamap, s_pamap = create_windows(
    pamap_data, pamap_labels, pamap_subjects
    )

    print(f"PAMAP windows: {X_pamap.shape}")

    # WISDM dataset
    X_wisdm, y_wisdm, s_wisdm = create_windows(
    wisdm_data, wisdm_labels, wisdm_subjects
    )

    print(f"WISDM windows: {X_wisdm.shape}")

    # -----------------------------------------------
    # STEP-5: Dataset Split (Pretrain vs Supervised)
    # -----------------------------------------------
    X_all = np.concatenate([X_pamap, X_wisdm])
    y_all = np.concatenate([y_pamap, y_wisdm])
    s_all = np.concatenate([s_pamap, s_wisdm])

    # Split 80% pretrain, 20% supervised
    split_idx = int(0.8 * len(X_all))

    X_pretrain = X_all[:split_idx]
    X_supervised = X_all[split_idx:]

    y_supervised = y_all[split_idx:]
    s_supervised = s_all[split_idx:]

    # ---------------------
    # STEP-6: Save Outputs
    # ---------------------
    np.save("data/processed/X_pretrain.npy", X_pretrain)

    np.save("data/processed/X_supervised.npy", X_supervised)
    np.save("data/processed/y_supervised.npy", y_supervised)
    np.save("data/processed/s_supervised.npy", s_supervised)

    # ------------------------------
    # STEP-7: Manifest File (.JSON)
    # ------------------------------
    import json

    manifest = {
        "datasets": ["PAMAP2", "WISDM"],
        "sampling_rate": 20,
        "window_size": 200,
        "step_size": 100,
        "channels": 6,
        "label_map": {
            "walking": 0,
            "running": 1,
            "sitting": 2,
            "standing": 3
        },
    "num_pretrain_samples": len(X_pretrain),
    "num_supervised_samples": len(X_supervised)
    }

    # Save Metadata (.JSON)
    with open("data/processed/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    # --------------------------
    # STEP-8: Validation Report
    # --------------------------
    print("\nFinal Summary:")
    print("Pretrain samples:", X_pretrain.shape)
    print("Supervised samples:", X_supervised.shape)
    print("Labels:", np.unique(y_supervised))

# -------------------------------
if __name__ == "__main__":
    process_har()
    

    