import pandas as pd
from pathlib import Path
from src.preprocessing.geocode import apply_geocoding


def load_and_clean_data(
    csv_path: Path,
    geocode_cache_path: Path,
    drop_columns: list,
    categorical_columns: list,
    numeric_columns: list
) -> pd.DataFrame:
    """
    Load and clean the CTTU dataset.
    
    Args:
        csv_path: Path to raw_dataset.csv
        geocode_cache_path: Path to geocode_cache.json
        drop_columns: Columns to remove
        categorical_columns: Categorical columns
        numeric_columns: Numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    print("LOADING AND CLEANING DATA")
    
    print(f"\nLoading: {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', engine='python')
    print(f"Loaded: {len(df):,} records")
    
    
    if 'data' in df.columns:
        df['DATA'] = df['DATA'].fillna(df['data'])
        df = df.drop(columns=['data'])
    
    df = df[df['DATA'].notna() & (df['DATA'] != '')]
    df.rename(columns={'DATA': 'Data'}, inplace=True)
    
    # Convert to datetime
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    
    print(f"Period: {df['Data'].min().date()} to {df['Data'].max().date()}")
    
    # TIME CLEANING    
    df['hora_dt'] = pd.to_datetime(df['hora'], format='%H:%M:%S', errors='coerce')
    df['hour'] = df['hora_dt'].dt.hour
    df['minute'] = df['hora_dt'].dt.minute
    df = df.drop(columns=['hora_dt'], errors='ignore')
    
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Duplicates removed: {removed:,}")

    df = df.drop(columns=drop_columns, errors='ignore')
    
    # CLEAN CATEGORICAL COLUMNS
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # REMOVE RECORDS WITHOUT ADDRESS
    
    before = len(df)
    df = df.dropna(subset=['endereco'])
    removed = before - len(df)
    if removed > 0:
        print(f" Records without address removed: {removed:,}")
    
    # GEOCODING  
    df = apply_geocoding(df)
    
    print(f"\n Final dataset: {len(df):,} records")
    
    return df

