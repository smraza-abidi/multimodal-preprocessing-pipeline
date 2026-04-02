import os
import urllib.request

# *************************
# Utility: Folder Creation
# *************************
def create_folders():
    folders = [
        "data/raw/PAMAP2",
        "data/raw/WISDM",
        "data/raw/EEGMMIDB",
        "data/raw/PTBXL"
        "data/interim",
        "data/processed"
        "sample_pack"
    ]

    for f in folders:
        os.makedirs(f, exist_ok=True)
        print(f"Folder ready: {f}")

# ***********************
# Utility: Data Download
# ***********************
def download_file(url, save_path):
    try:
        if not os.path.exists(save_path):
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, save_path)
            print(f"Saved to: {save_path}")
        else:
            print(f"Already exists: {save_path}")
    except Exception as e:
        print(f"[ERROR:] Failed to download {url}")
        print(e)

# *****************************
# Main Function logic download
# *****************************
def main():
    print("Setting up dataset folders...")
    
    # Calling function
    create_folders()

    print("\nStarting Dataset Download ...")
 
    # PAMAP2 placeholder download (as an example)
    pamap2_url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    pamap2_path = "data/raw/PAMAP2/pamap2+physical+activity+monitoring.zip"

    download_file(pamap2_url, pamap2_path)

    print("\nDownload Step Completed!")
    

# -------------------------------
if __name__ == "__main__":
    main()