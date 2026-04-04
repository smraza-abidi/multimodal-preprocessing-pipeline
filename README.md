### **Multimodal Time-Series Preprocessing Pipeline**

=====================================================

* ##### **Overview**



This repository implements a reproducible pipeline for downloading, organising, preprocessing, and validating multimodal time-series datasets including:



\* Human Activity Recognition (HAR): PAMAP2, WISDM, mHealth (Bonus)

\* EEG: EEGMMIDB

\* ECG: PTB-XL



The pipeline prepares data for downstream self-supervised learning workflows.



\-----------------------------------------------------------------------------



* ##### **Project Structure**



project/

|

|\_\_ README.md

|\_\_ setup\_data.sh

|

|\_\_ data/

|    └── raw/

|    └── interim/

|    └── processed/

|

|\_\_ scripts/

|    └── download.py

|    └── preprocess\_har.py

|    └── preprocess\_eeg.py

|    └── preprocess\_ecg.py

|    └── preprocess\_mhealth.py

|    └── create\_sample\_pack.py

|    └── validate\_outputs.py

|    └── test\_pipeline.py

|

|\_\_ reports/

|    └── one-page\_preprocessing\_plan.pdf

|    └── validation\_report.pdf

|    └── resource\_estimate.pdf

|

|\_\_ manifest/

|    └── har\_metadata.json

|    └── eeg\_metadata.json

|    └── ecg\_metadata.json

|    └── mhealth\_metadata.json

|

|\_\_ sample\_pack/

|    └── X\_har\_sample.npy

|    └── y\_har\_sample.npy

|    └── X\_eeg\_sample.npy

|    └── y\_eeg\_sample.npy

|    └── X\_ecg\_sample.npy

|    └── y\_ecg\_sample.npy

|    └── X\_mhealth\_sample.npy

|    └── y\_mhealth\_sample.npy

|

|\_\_ configs/

&#x20;    └── config.yaml





* ##### **Setup Instructions**



1. ###### **Create environment**



conda create -n multimodal python=3.11

conda activate multimodal

pip install numpy pandas wfdb mne pyyaml



\----------------------------------------

###### **2. Data setup**



python scripts/download.py



This Python script replaces the shell wrapper for dataset setup on non-Unix systems.

An equivalent shell wrapper (setup\_data.sh) is also provided. On Windows, pipeline

can be run directly via Python if bash is unavailable.



\----------------------------------------

###### **3. Datasets used**



* **HAR (PAMAP2 + WISDM):**

&#x20;  9 subjects were selected from the PAMAP2 and WISDM datasets for preprocessing and harmonisation.



* **HAR (mHealth) Bonus:**

&#x20;  A single subject was used from the mHealth dataset to demonstrate the preprocessing pipeline while

&#x20;  maintaining consistency and computational efficiency. The approach generalises to multiple subjects.



* **EEG (EEGMMIDB):**

&#x20;  1 subject was used with runs 4, 8, and 12 to perform event-based window extraction. A single

&#x20;  subject was used for EEG preprocessing to demonstrate the full pipeline, including annotation

&#x20;  parsing and event-aligned window extraction. The approach generalises to the full dataset.



* **ECG (PTB-XL):**

&#x20;  A subset of 30 patients was selected programmatically from the full PTB-XL metadata to ensure

&#x20;  reproducibility and manageable processing.



All subset selections are deterministic and can be reproduced using the provided preprocessing scripts.



\----------------------------------------

###### **4. Run preprocessing pipelines**



* **HAR:**

&#x09;python scripts/preprocess\_har.py



* **EEG:**

&#x20;  	python scripts/preprocess\_eeg.py



* **ECG:**

&#x20;  	python scripts/preprocess\_ecg.py



* **mhealth:**

&#x20;  	python scripts/preprocess\_mhealth.py





\# Sample pack:



python scripts/create\_sample\_pack.py



\-----------------------------------------

###### **5. Outputs**



* Processed data saved in:



&#x20;  data/processed/



**\*** (.npy) arrays for signals and labels



**HAR Dataset:**

X\_pretrain.npy

X\_supervised.npy

y\_supervised.npy



s\_supervised.npy



**EEG Dataset:**

X\_eeg.npy

y\_eeg.npy



**ECG Dataset:**

X\_ecg.npy

y\_ecg.npy



**mHealth Bonus Dataset:**

X\_mhealth.npy

y\_mhealth.npy





* Sample data saved in:



&#x20;  sample\_pack/



X\_har\_sample.npy

y\_har\_sample.npy

X\_eeg\_sample.npy

y\_eeg\_sample.npy

X\_ecg\_sample.npy

y\_ecg\_sample.npy

X\_mhealth\_sample.py

y\_mhealth\_sample.py



* Metadata files (.json) saved in:



&#x20;  manifest/



Metadata including subject ID, run ID, event type, and onset timing was preserved

and exported as JSON for traceability and downstream analysis.



har\_metadata.json

eeg\_metadata.json

ecg\_metadata.json

mhealth\_metadata.json



\-----------------------------------------

###### **6. Validation**



python scripts/validate\_outputs.py



Run basic tests:

python scripts/test\_pipeline.py



\-----------------------------------------

###### **7. Reproducibility**



Run the above commands sequentially from a clean project directory to fully reproduce the pipeline.

For efficiency and to meet assessment constraints, the pipeline processes a representative subset

while remaining fully scalable to the complete dataset.



All pipelines:



\* Use fixed preprocessing steps

\* Maintain consistent sampling rates

\* Preserve metadata (subject, labels, timestamps)



\-----------------------------------------

###### **Optional: Self-Supervised Learning Perspective**



The harmonised multimodal HAR datasets (PAMAP2, WISDM, mHealth) can be used in a self-supervised learning setup.



\- Pretraining: contrastive learning on unlabeled windows (e.g., augmentations such as noise, scaling, masking)

\- Encoder: shared temporal model (e.g., CNN or Transformer)

\- Fine-tuning: supervised classification using labelled subsets



This enables robust representation learning across heterogeneous sensor modalities.



\-----------------------------------------

###### **Data Location:**



All the processed data and sample pack are available on the following link:



Google drive link: https://drive.google.com/drive/folders/1c4U9SEiVBZB3Xj_5rRNaerpH1yWj2q3p?usp=sharing

###### 

###### **Notes:**



\* Subsets of datasets were used for efficiency

\* EEG uses event-aligned windows (T1/T2)

\* Patient-level split applied for ECG

\* mHealth dataset included as an optional extension, aligned to the same 20 Hz HAR schema



\------------------------------------------

