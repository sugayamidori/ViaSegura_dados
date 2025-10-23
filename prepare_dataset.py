import pandas as pd
from pathlib import Path
from src.preprocessing.data_loader import load_and_clean_data
from src.preprocessing.temporal_features import create_temporal_features
from src.preprocessing.grid import add_jitter, add_h3
from sklearn.preprocessing import LabelEncoder

from src.config.config import (
    RAW_DATASET_PATH,
    GEOCODE_CACHE_PATH,
    PROCESSED_DATASET_PATH,
    DROP_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_PERIODS,
    MUNICIPAL_HOLIDAYS
)

def prepare_processed_dataset():
    """
    Prepara o dataset COMPLETAMENTE processado e pronto para treino.
    
    Pipeline completo:
    1. Load and clean (já inclui geocoding)
    2. Temporal features
    3. Add jitter
    4. Add H3 cells
    5. Save
    """
    print("PREPARING FULLY PROCESSED DATASET")
   
    

    print("\n[1/4] Loading and cleaning data (includes geocoding)...")
    df = load_and_clean_data(
        csv_path=RAW_DATASET_PATH,
        geocode_cache_path=GEOCODE_CACHE_PATH,
        drop_columns=DROP_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS
    )
    print(f"Records after cleaning: {len(df):,}")
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        with_coords = df[['latitude', 'longitude']].notna().all(axis=1).sum()
        print(f"Records with coordinates: {with_coords:,} ({with_coords/len(df)*100:.1f}%)")
    
    print("\n[2/4] Creating temporal features...")
    df = create_temporal_features(
        df=df,
        time_periods=TIME_PERIODS,
        municipal_holidays=MUNICIPAL_HOLIDAYS
    )
    temporal_cols = [
        'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
        'quarter', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos', 'weekend', 'holiday'
    ]
    created_temporal = len([c for c in temporal_cols if c in df.columns])
    print(f"Temporal features created: {created_temporal}")
    
    print("\n[3/4] Adding jitter to addresses without number...")
    df = add_jitter(df)
    print(f"Jitter applied")
    
    print("\n[4/4] Adding H3 spatial index...")
    df = add_h3(df)
    h3_count = df['h3_cell'].notna().sum()
    print(f"Records with H3 cells: {h3_count:,} ({h3_count/len(df)*100:.1f}%)")
    
    # Encode the CLEANED bairro version (more reliable)
    if 'bairro_clean' in df.columns:
        le = LabelEncoder()
        df['bairro_encoded'] = le.fit_transform(
            df['bairro_clean'].fillna('DESCONHECIDO').astype(str)
        )
        print(f"'bairro_encoded' created from 'bairro_clean' ({len(le.classes_)} bairros únicos)")
    elif 'bairro' in df.columns:
        # Fallback to raw 'bairro' if clean version doesn't exist
        le = LabelEncoder()
        df['bairro_encoded'] = le.fit_transform(
            df['bairro'].fillna('DESCONHECIDO').astype(str)
        )
        print(f" Used raw 'bairro' for encoding ({len(le.classes_)} bairros únicos)")
    else:
        df['bairro_encoded'] = 0
        print("Neither 'bairro_clean' nor 'bairro' found. 'bairro_encoded' = 0.")
        
    # FINAL: Save
    processed_dir = Path(PROCESSED_DATASET_PATH).parent
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    print("DATASET PROCESSADO E SALVO COM SUCESSO!")
   
    print(f"Arquivo: {PROCESSED_DATASET_PATH}")
    print(f"Registros: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")
    
    # Estatísticas de qualidade
    print(f"\nQualidade dos Dados:")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        coords_ok = df[['latitude', 'longitude']].notna().all(axis=1).sum()
        print(f"  Coordenadas: {coords_ok:,} ({coords_ok/len(df)*100:.1f}%)")
    if 'h3_cell' in df.columns:
        h3_ok = df['h3_cell'].notna().sum()
        print(f"  H3 cells: {h3_ok:,} ({h3_ok/len(df)*100:.1f}%)")
    
    # Preview das colunas importantes
    print(f"\nColunas Principais:")
    important_cols = ['Data', 'hora', 'latitude', 'longitude', 'h3_cell', 
                      'year', 'month', 'day_of_week', 'weekend', 'holiday',
                      'bairro', 'endereco', 'tipo']
    available = [c for c in important_cols if c in df.columns]
    print(f"  Disponíveis: {', '.join(available[:10])}")
    if len(available) > 10:
        print(f"  ... e mais {len(available)-10} colunas")
    
    
    return df


if __name__ == "__main__":
    df = prepare_processed_dataset()
    print("\Done! Now you can run: python main.py")