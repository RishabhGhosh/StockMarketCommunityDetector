#!/bin/bash

# Install all the prerequisite
sudo apt-get update
sudo apt-get install -y git python3-pip

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r ./requirements.txt