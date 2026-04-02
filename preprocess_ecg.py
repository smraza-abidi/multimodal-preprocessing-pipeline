import os
import numpy as np
import pandas as pd
import wfdb
import json
import ast

# ****************************
# Load Metadata (PTB-XL .csv)
# ****************************
def load_ptbxl_metadata(csv_path):

    df = pd.read_csv(csv_path)

    # First 30 Patients (from ptbxl_database.csv)
    selected_patients = [15709, 13243, 20372, 17014, 17448, 19005, 16193, 11275, 18792, 9456, 11243, 11031,
                         19953, 12925, 13375, 10999, 13619, 11116, 17102, 20978, 9012, 10962, 10316, 14340,
                         19053, 15348, 11154, 20527, 15539, 8787]
    
    df = df[df['patient_id'].isin(selected_patients)]

    print("Metadata loaded:", df.shape)

    return df

# ****************************
# Extract Label
# ****************************
def extract_label(label_str):

    label_dict = ast.literal_eval(label_str)

    if len(label_dict) > 0:
        return list(label_dict.keys())[0]   # take first label
    else:
        return "unknown"

# ****************************
# Load ECG Signal
# ****************************
def load_ecg_signal(base_path, filename):

    record_path = os.path.join(base_path, filename)

    signal, meta = wfdb.rdsamp(record_path)

    return signal   # shape: (time, leads)

# ****************************
# Windowing
# ****************************
def create_windows(data, labels, subjects, window_size=200, step_size=100):

    X, y, s = [], [], []

    for i in range(0, len(data) - window_size, step_size):

        window = data[i:i+window_size]

        label = 0  # dummy for ECG
        subject = 0

        X.append(window)
        y.append(label)
        s.append(subject)

    return np.array(X), np.array(y), np.array(s)

# ****************************
# ECG Processing
# ****************************
def process_ecg(data_dir):

    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = load_ptbxl_metadata(csv_path)

    print("\nProcessing ECG (PTB-XL) ...")

    X_all = []
    y_all = []
    meta_all = []

    # filter records (30-patients)
    available_files = set([
        f.replace(".dat", "") for f in os.listdir(os.path.join(data_dir, "records100"))
        if f.endswith(".dat")
    ])

    # Extract only filename from CSV
    df['file_key'] = df['filename_lr'].astype(str).apply(lambda x: x.split('/')[-1])

    # DEBUG PRINT
    # print("\nSample CSV keys:", df['file_key'].head(5).tolist())
    # print("Sample local files:", list(available_files)[:5])

    # df = df[df['filename_lr'].isin(available_files)]

    print("Filtered records:", len(df))

    for _, row in df.iterrows():

        file_key = row['filename_lr'].split('/')[-1]
        file_path = os.path.join(data_dir, "records100", file_key)

        if not os.path.exists(file_path + ".dat"):
            continue

        try:
            signal = load_ecg_signal(
                os.path.join(data_dir, "records100"),
                file_key
            )

            label = extract_label(row['scp_codes'])

            # Windowing (reuse HAR function)
            dummy_labels = np.zeros(len(signal))
            dummy_subjects = np.zeros(len(signal))

            X, _, _ = create_windows(signal, dummy_labels, dummy_subjects)

            X_all.append(X)
            y_all.extend([label] * len(X))

            # Metadata
            for _ in range(len(X)):
                meta_all.append({
                    "patient_id": int(row['patient_id']),
                    "record_id": int(row['ecg_id']),
                    "label": label,
                    "sampling_rate": 100,
                    "leads": signal.shape[1]
                })

        except Exception as e:
            print("Skipping:", row['filename_lr'], e)

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)

    print("ECG shape:", X_all.shape)

    return X_all, y_all, meta_all, df

# ****************************
# Patient-level Split
# ****************************
def patient_split(df, test_size=0.2):

    patients = df['patient_id'].unique()

    np.random.shuffle(patients)

    split_idx = int(len(patients) * (1 - test_size))

    train_patients = patients[:split_idx]
    test_patients = patients[split_idx:]

    print("Train patients:", len(train_patients))
    print("Test patients:", len(test_patients))

    return train_patients, test_patients

# ****************************
# Main Function
# ****************************
if __name__ == "__main__":

    data_dir = "data/raw/PTBXL"

    X_ecg, y_ecg, meta_ecg, df = process_ecg(data_dir)

    # Patient-level split
    print("\nTraining and Testing Patients:")
    train_patients, test_patients = patient_split(df)

    print("\nECG pipeline completed!")
    print("ECG shape:", X_ecg.shape)
    print("Unique labels:", np.unique(y_ecg))

    # # Save Outputs
    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_ecg.npy", X_ecg)
    np.save("data/processed/y_ecg.npy", y_ecg)

    # Save Metadata (.JSON)
    with open("data/processed/ecg_metadata.json", "w") as f:
        json.dump(meta_ecg, f, indent=4)
