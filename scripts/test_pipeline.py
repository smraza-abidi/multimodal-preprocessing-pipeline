import numpy as np

def test_har_shape():
    X = np.load("data/processed/X_supervised.npy", allow_pickle=True)
    assert X.shape[1] == 200, "HAR window size incorrect"

def test_eeg_shape():
    X = np.load("data/processed/X_eeg.npy", allow_pickle=True)
    assert X.shape[1] > 0, "EEG data empty"

def test_ecg_shape():
    X = np.load("data/processed/X_ecg.npy", allow_pickle=True)
    assert X.shape[-1] == 12, "ECG leads mismatch"

def test_mhealth_shape():
    X = np.load("data/processed/X_mhealth.npy", allow_pickle=True)
    assert X.shape[1] == 200, "mHealth window size incorrect"

def run_tests():
    test_har_shape()
    test_eeg_shape()
    test_ecg_shape()
    test_mhealth_shape()
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
