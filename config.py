import os
from pathlib import Path

# GitHub repository configurations
REPO_NAME = "medical-image-analysis"
DATASET_URL = "https://github.com/datascintist-abusufian/medical-image-analysis/raw/main/images.zip"
DATASET_FILENAME = "images.zip"
EXTRACT_FOLDER = "images"

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset paths
DATASET_PATH = DATA_DIR / DATASET_FILENAME
EXTRACT_PATH = DATA_DIR / EXTRACT_FOLDER
