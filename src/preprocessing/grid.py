import numpy as np
import pandas as pd
import h3
from src.config.config import H3_RESOLUTION, SIGMA_METERS

def add_jitter(df):
    sigma_lat = SIGMA_METERS / 111000
    sigma_lon = SIGMA_METERS / 110000
    mask_no_number = (df['numero_clean'] == '') | (df['numero'] == 'unknown')
    mask_apply = mask_no_number & df['latitude'].notna()
    n_apply = mask_apply.sum()
    if n_apply > 0:
        np.random.seed(42)
        df.loc[mask_apply, 'latitude'] += np.random.normal(0, sigma_lat, n_apply)
        df.loc[mask_apply, 'longitude'] += np.random.normal(0, sigma_lon, n_apply)
    return df

def add_h3(df):
    def lat_lon_to_h3(row):
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            try:
                return h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION)
            except Exception:
                return None
        return None
    df['h3_cell'] = df.apply(lat_lon_to_h3, axis=1)
    return df
