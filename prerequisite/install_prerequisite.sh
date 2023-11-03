#!/bin/bash

# Install all the prerequisite
sudo apt-get update
sudo apt-get install -y git python3-pip

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Add packages to requirements.txt
echo "holidays" >> requirements.txt
echo "yfinance" >> requirements.txt

# Install the required packages
pip install -r ./requirements.txt

# For the finBERT requirement
# Define the required Python packages
packages=(
    "jupyter>=1.0.0"
    "pandas>=0.23.4"
    "numpy>=1.16.3"
    "nltk"
    "tqdm"
    "ipykernel>=5.1.3"
    "transformers>=4.1.1"
    "joblib>=0.13.2"
    "scikit-learn>=0.21.2"
    "spacy>=2.1.4"
    "torch>=1.1.0"
    "textblob"
)

# Install the packages using pip
pip install "${packages[@]}"
