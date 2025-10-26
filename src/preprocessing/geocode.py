import json
import pandas as pd
from src.utils import clean_address_part
from src.config.config import GEOCODE_CACHE_PATH

def apply_geocoding(df):
    df['endereco_clean'] = df['endereco'].apply(clean_address_part)
    df['numero_clean'] = df['numero'].apply(clean_address_part)
    df['bairro_clean'] = df['bairro'].apply(clean_address_part)

    df['address_to_geocode'] = (
        df['endereco_clean'] + ", " +
        df['numero_clean'] + ", " +
        df['bairro_clean'] + ", Recife, Pernambuco, Brasil"
    ).str.replace(", ,", ",").str.replace(" ,", ",").str.strip(", ")

    with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    coords_df = pd.DataFrame(
        [(addr, coords[0], coords[1]) for addr, coords in cache.items()],
        columns=["address_to_geocode", "latitude", "longitude"]
    )

    df = df.merge(coords_df, on="address_to_geocode", how="left")
    with_coords = df['latitude'].notna().sum()
    without_coords = df['latitude'].isna().sum()

    print(f"With coordinates: {with_coords:,}")
    print(f"Without coordinates: {without_coords:,}")
    
    return df
