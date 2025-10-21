from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
BACKEND_EXPORT_DIR = PROJECT_ROOT / "backend_export"

# Files
MERGED_DATASET_PATH = RAW_DATA_DIR / "merged_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_dataset.csv"
GEOCODE_CACHE_PATH = RAW_DATA_DIR / "geocode_cache.json"

MODEL_CONFIG = {
    'objective': 'poisson',
    'metric': 'poisson',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}
N_SPLITS = 5
N_BOOST_ROUNDS = 200
PREDICTION_WEEKS = 12

# H3
H3_RESOLUTION = 9  # ~50m hexagons

SIGMA_METERS = 30

# Features
FEATURE_COLUMNS = [
    'year', 'month', 'week_of_year', 'holiday', 'weekend',
    'month_sin', 'month_cos', 'week_sin', 'week_cos',
    'sinistros_media_4w', 'sinistros_media_12w',
    'sinistros_lag_1w', 'sinistros_lag_4w',
    'total_historico_celula',
    'auto_historico', 'moto_historico', 
    'onibus_historico', 'caminhao_historico',
    'bairro_clean_encoded'
]

PANDEMIC_YEARS = [2020, 2021]