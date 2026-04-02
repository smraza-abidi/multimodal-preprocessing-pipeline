import mne
import numpy as np
import os

# ************************************
# STEP-1: Load EDF and extract events
# ************************************ 
def load_eeg_events(file_path):

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Filtering (noise remove)
    raw.filter(1, 40, verbose=False)
    
    # Re-referencing
    raw.set_eeg_reference('average', verbose=False)

    raw.resample(20) # Resampling 160 Hz -> 20 Hz 
    sfreq = 20

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw)

    sfreq = raw.info['sfreq']

    return raw, events, event_id, sfreq

# **********************************
# STEP-2: Create event based window
# ********************************** 
def extract_eeg_windows(raw, events, event_id, sfreq, subject_id, run_id):

    X = []
    y = []
    meta_list = []

    window_samples = int(4 * sfreq)  # 4 seconds

    for event in events:

        onset_sample = event[0]
        event_code = event[2]

        # Only T1 and T2
        if event_code == event_id.get('T1'):
            label = 0  # left fist
        elif event_code == event_id.get('T2'):
            label = 1  # right fist
        else:
            continue

        start = onset_sample
        end = start + window_samples

        if end > raw.n_times:
            continue

        # Extract window
        window = raw.get_data(start=start, stop=end).T  # (time, channels)

        X.append(window)
        y.append(label)

        meta = {
            "subject": subject_id,
            "run": run_id,
            "event": int(event_code),
            "onset": int(onset_sample)
        }
        meta_list.append(meta)

    return np.array(X), np.array(y), meta_list


# Debug Check run 4,8,12
def is_valid_run(filename):
    return any(r in filename for r in ["R04", "R08", "R12"])


# *****************************
# STEP-3: Process EEG Pipeline
# ***************************** 
def process_eeg(data_dir):

    X_all = []
    y_all = []
    meta_all = []

    print("\nProcessing EEG (EEGMMIDB) ...")

    for file in os.listdir(data_dir):

        if file.endswith(".edf") and is_valid_run(file):

            file_path = os.path.join(data_dir, file)

            raw, events, event_id, sfreq = load_eeg_events(file_path)

            subject_id = int(file[1:4])
            run_id = int(file[5:7])

            X, y, meta = extract_eeg_windows(raw, events, event_id, sfreq, subject_id, run_id)

            X_all.append(X)
            y_all.append(y)

            print(f"{file} → {X.shape}")

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    meta_all.extend(meta)

    print(f"Final EEG shape: {X_all.shape}")

    return X_all, y_all, meta_all

# **************************
# Main Function
# **************************
if __name__ == "__main__":

    eeg_dir = "data/raw/EEGMMIDB/S001"

    X_eeg, y_eeg, meta_eeg = process_eeg(eeg_dir)

    print("\nEEG pipeline completed")
    print("EEG data shape:", X_eeg.shape)
    print("EEG labels:", np.unique(y_eeg))

    # print("Metadata length:", len(meta_eeg))
    # print("Sample metadata:", meta_eeg[:3])

    # Save Metadata (.JSON)
    import json
    os.makedirs("manifest", exist_ok=True)

    with open("manifest/eeg_metadata.json", "w") as f:
         json.dump(meta_eeg, f, indent=4)
    
    # Save NumPy Files
    np.save("data/processed/X_eeg.npy", X_eeg)
    np.save("data/processed/y_eeg.npy", y_eeg)
    
