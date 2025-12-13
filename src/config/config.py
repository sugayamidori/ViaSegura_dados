from pathlib import Path

# GPU/CUDA CONFIGURATION
# IMPORTANT: This project supports GPU acceleration via CUDA
# To enable:
# 1. Install: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
# 2. Change USE_GPU to True
# 3. Expected speedup: 5-10x faster training

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
BACKEND_EXPORT_DIR = PROJECT_ROOT / "backend_export"
PROCESSED_DIR = PROJECT_ROOT / "processed"

# Files
GEOCODE_FULL_COMPLETE = PROCESSED_DIR / "geocoded_complete.csv"
RECIFE_H3_GRID_PATH = RAW_DATA_DIR /  "grid_recife_h3_res9.csv"

WEEKLY_DATASET_PATH = PROCESSED_DIR / "weekly_agreggation_dataset.csv"
PANDEMIC_YEARS = [2020, 2021]  

USE_GPU = False  # True = GPU | False = CPU
GPU_DEVICE_ID = 0  # ID da GPU (0 = first GPU, 1 = second, etc)

MODEL_CONFIG = {
    'objective': 'poisson',
    'metric': 'poisson',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

N_SPLITS = 5  
N_BOOST_ROUNDS = 200
