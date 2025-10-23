from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
BACKEND_EXPORT_DIR = PROJECT_ROOT / "backend_export"
PROCESSED_DIR = PROJECT_ROOT / "processed"

# Files
RAW_DATASET_PATH = RAW_DATA_DIR / "raw_dataset.csv"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "processed_dataset.csv"
GEOCODE_CACHE_PATH = RAW_DATA_DIR / "geocode_cache.json"

PANDEMIC_YEARS = [2020, 2021]  

# GPU/CUDA CONFIGURATION
# IMPORTANT: This project supports GPU acceleration via CUDA
# To enable:
# 1. Install: pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
# 2. Change USE_GPU to True
# 3. Expected speedup: 5-10x faster training

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
PREDICTION_WEEKS = 12 

H3_RESOLUTION = 9  # ~174m edge length
SIGMA_METERS = 30  # Jitter para endereços sem número

FEATURE_COLUMNS = [
    
    'year', 'month', 'week_of_year', 'holiday', 'weekend',
    'month_sin', 'month_cos', 'week_sin', 'week_cos',
    
    'sinistros_media_4w', 'sinistros_media_12w',
    'sinistros_lag_1w', 'sinistros_lag_4w',
    'total_historico_celula',
    
    'auto_historico', 'moto_historico', 
    'onibus_historico', 'caminhao_historico',
    
    'bairro_encoded'
]

DROP_COLUMNS = [
    'detalhe_endereco_acidente', 'numero_cruzamento', 'num_semaforo',
    'sentido_via', 'acidente_verificado', 'tempo_clima', 'situacao_semaforo',
    'sinalizacao', 'condicao_via', 'conservacao_via', 'ponto_controle',
    'situacao_placa', 'velocidade_max_via', 'mao_direcao', 'divisao_via1',
    'divisao_via2', 'divisao_via3', 'Protocolo'
]

CATEGORICAL_COLUMNS = [
    'natureza_acidente', 'situacao', 'bairro', 'endereco',
    'numero', 'complemento', 'endereco_cruzamento',
    'referencia_cruzamento', 'bairro_cruzamento', 'tipo', 'descricao'
]

NUMERIC_COLUMNS = [
    'auto', 'moto', 'ciclom', 'ciclista', 'pedestre',
    'onibus', 'caminhao', 'viatura', 'outros',
    'vitimas', 'vitimasfatais'
]

MUNICIPAL_HOLIDAYS = {
    "Carta Magna": (3, 6),
    "São João": (6, 24),
    "Nossa Senhora do Carmo": (7, 16),
    "Nossa Senhora da Conceição": (12, 8)
}

TIME_PERIODS = {
    'morning_rush': (6, 9),
    'evening_rush': (17, 20),
    'night': (20, 6),
    'business_hours': (8, 18)
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

TEMPORAL_FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'day_of_year', 
    'week_of_year', 'quarter', 'hour', 'minute',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos',
    'weekend', 'holiday'
]

SPATIAL_FEATURES = [
    'latitude', 'longitude', 'h3_cell',
    'endereco', 'bairro', 'numero'
]

CYCLIC_FEATURES = [
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
]
VEHICLE_COLUMNS = ['auto', 'moto', 'onibus', 'caminhao']
VICTIM_COLUMNS = ['vitimas', 'vitimasfatais']