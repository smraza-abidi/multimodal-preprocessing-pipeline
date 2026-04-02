import numpy as np
import pandas as pd
import os

os.makedirs("sample_pack", exist_ok=True)

# HAR
X_har = np.load("data/processed/X_supervised.npy", allow_pickle=True)
np.save("sample_pack/X_har_sample.npy", X_har[:100])

y_har = np.load("data/processed/y_supervised.npy", allow_pickle=True)
np.save("sample_pack/y_har_sample.npy", y_har[:100])

# EEG
X_eeg = np.load("data/processed/X_eeg.npy", allow_pickle=True)
np.save("sample_pack/X_eeg_sample.npy", X_eeg[:100])

y_eeg = np.load("data/processed/y_eeg.npy", allow_pickle=True)
np.save("sample_pack/y_eeg_sample.npy", y_eeg[:100])

# ECG
X_ecg = np.load("data/processed/X_ecg.npy", allow_pickle=True)
np.save("sample_pack/X_ecg_sample.npy", X_ecg[:100])

y_ecg = np.load("data/processed/y_ecg.npy", allow_pickle=True)
np.save("sample_pack/y_ecg_sample.npy", y_ecg[:100])

print("Sample pack created!")
