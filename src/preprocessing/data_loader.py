import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Tuple

def load_and_clean_data(
    csv_path: Path,
    geocode_cache_path: Path,
    drop_columns: list,
    categorical_columns: list,
    numeric_columns: list
) -> pd.DataFrame:
    """
    Load and clean CTTU dataset.
    
    Args:
        csv_path: Path to merged_dataset.csv
        geocode_cache_path: Path to geocode_cache.json
        drop_columns: Columns to drop
        categorical_columns: Categorical column names
        numeric_columns: Numeric column names
    
    Returns:
        Cleaned DataFrame
    """
    print(f"\n Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', engine='python')
    print(f"Loaded {len(df):,} records")
    
        
    
    df['DATA'] = df['DATA'].fillna(df['data'])
    df = df[df['DATA'].notna() & (df['DATA'] != '')]
    df.drop('data', axis=1, inplace=True)
    df.rename(columns={'DATA':'Data'}, inplace=True)

    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    df = df.drop_duplicates()
    
    # Clean dates
    if 'data' in df.columns:
        df['Data'] = df['Data'].fillna(df['data'])
        df = df.drop(columns=['data'], errors='ignore')
    
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    print(f"Date range: {df['Data'].min().date()} to {df['Data'].max().date()}")
    
    # Parse time
    df['hora_dt'] = pd.to_datetime(df['hora'], format='%H:%M:%S', errors='coerce')
    df['hour'] = df['hora_dt'].dt.hour
    df['minute'] = df['hora_dt'].dt.minute
    df = df.drop(columns=['hora_dt'], errors='ignore')
    
    # Drop unnecessary columns
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # Clean categorical
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Clean numeric
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.dropna(subset=['endereco'])
    
    # Geocoding
    df = _add_geocoding(df, geocode_cache_path)
    
    print(f"\nFinal dataset: {len(df):,} records")
    print("="*70)
    
    return df


def _add_geocoding(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Add geocoding from cache."""
    
    def clean_address(x):
        if pd.isna(x):
            return ""
        x = str(x).strip().lower()
        if x in ["", "nan", "none", "unknown"]:
            return ""
        # Expand abbreviations
        abbreviations = {
            r'\bav\b': 'avenida', r'\br\b': 'rua',
            r'\bestr\b': 'estrada', r'\best\b': 'estrada',
            r'\brod\b': 'rodovia', r'\btrav\b': 'travessa',
            r'\bal\b': 'alameda', r'\bpça\b': 'praça'
        }
        for abbr, full in abbreviations.items():
            x = re.sub(abbr, full, x)
        return re.sub(r'\s+', ' ', x).strip()
    
    df['endereco_clean'] = df['endereco'].apply(clean_address)
    df['numero_clean'] = df['numero'].apply(clean_address)
    df['bairro_clean'] = df['bairro'].apply(clean_address)
    
    # Create address for geocoding
    df['address_to_geocode'] = (
        df['endereco_clean'] + ", " +
        df['numero_clean'] + ", " +
        df['bairro_clean'] + ", Recife, Pernambuco, Brasil"
    ).str.replace(", ,", ",").str.replace(" ,", ",").str.strip(", ")
    
    # Remove empty addresses
    df = df[df['endereco_clean'] != ""]
    
    # Load cache
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    
    coords_df = pd.DataFrame(
        [(addr, coords[0], coords[1]) for addr, coords in cache.items()],
        columns=["address_to_geocode", "latitude", "longitude"]
    )
    
    df = df.merge(coords_df, on="address_to_geocode", how="left")
    
    
    return df