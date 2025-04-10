import os

# Get the absolute path to the current file (i.e., path_config.py)
CURRENT_FILE = os.path.abspath(__file__)

# Go up to the root project directory (adjust levels as needed)
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))  # Up 2 levels

# Define folders
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")
DEPENDENCIES_DIR = os.path.join(PROJECT_DIR, "dependencies")
