#!/bin/bash

echo "Creating project directories..."

mkdir -p data/raw/PAMAP2
mkdir -p data/raw/WISDM
mkdir -p data/raw/EEGMMIDB
mkdir -p data/raw/PTBXL
mkdir -p data/interim
mkdir -p data/processed
mkdir -p sample_pack

echo "Directories created."

echo "Starting dataset download..."

python scripts/download.py

echo "Setup complete."