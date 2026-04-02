import numpy as np
import os

def check_file(path):
    if not os.path.exists(path):
        print(f"[Error] Missing file: {path}")
        return False
    print(f"\n[OK] Found: {path}")
    return True


def check_array(name, arr):
    print(f"\nChecking {name}...")

    print("Shape:", arr.shape)

    try:
        if np.isnan(arr.astype(float)).any():
            print("[ERROR] NaN values found!")
        else:
            print("[OK] No NaNs")
    except:
        print("[WARN] Cannot check NaNs (non-numeric array)")

    try:
        if np.isinf(arr.astype(float)).any():
            print("[ERROR] Inf values found!")
        else:
            print("[OK] No Infs")
    except:
        print("[WARN] Cannot check Infs (non-numeric array)")


def main():

    print("=== VALIDATION START ===")

    # ---------------- HAR ----------------
    if check_file("data/processed/X_supervised.npy"):
        X_har = np.load("data/processed/X_supervised.npy", allow_pickle=True)
        y_har = np.load("data/processed/y_supervised.npy", allow_pickle=True)

        check_array("HAR X", X_har)
        print("HAR labels:", np.unique(y_har))

    # ---------------- EEG ----------------
    if check_file("data/processed/X_eeg.npy"):
        X_eeg = np.load("data/processed/X_eeg.npy", allow_pickle=True)
        y_eeg = np.load("data/processed/y_eeg.npy", allow_pickle=True)

        check_array("EEG X", X_eeg)
        print("EEG labels:", np.unique(y_eeg))

    # ---------------- ECG ----------------
    if check_file("data/processed/X_ecg.npy"):
        X_ecg = np.load("data/processed/X_ecg.npy", allow_pickle=True)
        y_ecg = np.load("data/processed/y_ecg.npy", allow_pickle=True)

        check_array("ECG X", X_ecg)
        print("ECG labels:", np.unique(y_ecg))

    # ---------------- mhealth ----------------
    if check_file("data/processed/X_mhealth.npy"):
        X_m = np.load("data/processed/X_mhealth.npy", allow_pickle=True)
        y_m = np.load("data/processed/y_mhealth.npy", allow_pickle=True)

        check_array("mHealth X", X_m)
        print("mHealth labels:", np.unique(y_m))


    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
