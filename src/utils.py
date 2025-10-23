import re
import pandas as pd
import numpy as np
from typing import Optional, List

def clean_address_part(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if x in ["", "nan", "none", "unknown", "desconhecido"]:
        return ""
    abbreviations = {
        r'\bav\b': 'avenida', r'\br\b': 'rua', r'\bestr\b': 'estrada',
        r'\best\b': 'estrada', r'\brod\b': 'rodovia', r'\btrav\b': 'travessa',
        r'\bal\b': 'alameda', r'\bpça\b': 'praça'
    }
    for abbr, full in abbreviations.items():
        x = re.sub(abbr, full, x)
    x = re.sub(r'\s+', ' ', x)
    return x
def safe_mode(x):
    """Return the most frequent value; fallback to first if no mode exists."""
    if len(x) == 0:
        return 'unknown'
    mode_result = x.mode()
    return mode_result.iloc[0] if len(mode_result) > 0 else x.iloc[0]

def aggregate_weekly_by_h3(
    df: pd.DataFrame,
    h3_column: str = 'h3_cell',
    date_column: str = 'Data',
    pandemic_years: Optional[List[int]] = None,
    vehicle_columns: Optional[List[str]] = None,
    victim_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate accident records by H3 cell and week, ensuring a complete grid
    (all cells × all weeks in the period), filling with zeros where no accidents occurred.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing date, H3, vehicle, victim, and metadata columns.
    h3_column : str, optional
        Name of the H3 index column (default: 'h3_cell').
    date_column : str, optional
        Name of the date column (default: 'Data').
    pandemic_years : list of int, optional
        Years to exclude (e.g., [2020, 2021]). If None, no filtering is applied.
    vehicle_columns : list of str, optional
        Columns representing vehicle counts (e.g., ['auto', 'moto', ...]).
    victim_columns : list of str, optional
        Columns representing victim counts (e.g., ['vitimas', 'vitimasfatais']).
    categorical_columns : list of str, optional
        Categorical columns to aggregate using mode (e.g., ['bairro_clean']).

    Returns
    -------
    pd.DataFrame
        Weekly-aggregated DataFrame with complete H3 × week grid and zero-filled missing weeks.
    """
    
    print("WEEKLY AGGREGATION BY H3 CELL")
    

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    if pandemic_years:
        print(f"\n[PRE-FILTER] Removing pandemic years: {pandemic_years}")
        before = len(df)
        df = df[~df[date_column].dt.year.isin(pandemic_years)]
        print(f"  - Removed records: {before - len(df):,}")
        print(f"  - Remaining records: {len(df):,}")

    df_h3 = df[df[h3_column].notna()].copy()
    print(f"\nRecords with H3: {len(df_h3):,}")

    if len(df_h3) == 0:
        raise ValueError("No valid H3 cells found in the dataset.")

    # Add ISO year-week identifier and week start date
    df_h3['year_week'] = df_h3[date_column].dt.isocalendar().year * 100 + \
                         df_h3[date_column].dt.isocalendar().week
    df_h3['week_start'] = df_h3[date_column].dt.to_period('W').apply(lambda r: r.start_time)

    if vehicle_columns is None:
        vehicle_columns = ['auto', 'moto', 'onibus', 'caminhao']
    if victim_columns is None:
        victim_columns = ['vitimas', 'vitimasfatais']
    if categorical_columns is None:
        categorical_columns = ['bairro_clean']

    # Define aggregation rules
    agg_dict = {
        date_column: 'count',
        'holiday': 'max',
        'weekend': 'max',
        'month': 'first',
        'year': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'bairro_encoded': 'first'
    }

    # Add cyclic features if present
    cyclic_cols = ['month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
    for col in cyclic_cols:
        if col in df_h3.columns:
            agg_dict[col] = 'first'

    # Add vehicle and victim columns (sum)
    for col in vehicle_columns + victim_columns:
        if col in df_h3.columns:
            agg_dict[col] = 'sum'

    # Add categorical columns (safe mode)
    for col in categorical_columns:
        if col in df_h3.columns:
            agg_dict[col] = safe_mode

    df_sinistros = df_h3.groupby([h3_column, 'year_week', 'week_start']).agg(agg_dict).reset_index()
    df_sinistros.rename(columns={date_column: 'num_sinistros'}, inplace=True)

    print(f"\n Weeks with accidents: {len(df_sinistros):,} records")

    # Build complete grid: all cells × all weeks
    unique_cells = df_h3[h3_column].unique()
    unique_weeks = df_h3[['year_week', 'week_start']].drop_duplicates().sort_values('year_week')

    print(f"\n Building complete grid:")
    print(f"  - Unique H3 cells: {len(unique_cells)}")
    print(f"  - Unique weeks: {len(unique_weeks)}")

    complete_grid = pd.merge(
        pd.DataFrame({h3_column: unique_cells}),
        unique_weeks,
        how='cross'
    )
    print(f"  - Total combinations: {len(complete_grid):,}")

    # Merge real data into the full grid
    df_agg = complete_grid.merge(
        df_sinistros,
        on=[h3_column, 'year_week', 'week_start'],
        how='left'
    )

    df_agg['num_sinistros'] = df_agg['num_sinistros'].fillna(0)

    # Reconstruct year/month from week_start where missing
    df_agg['year'] = df_agg['year'].fillna(df_agg['week_start'].dt.year)
    df_agg['month'] = df_agg['month'].fillna(df_agg['week_start'].dt.month)
    df_agg['week_of_year'] = df_agg['week_start'].dt.isocalendar().week

    # Fill spatial and categorical metadata using cell-level statistics
    cell_metadata = df_h3.groupby(h3_column).agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    for col in categorical_columns:
        if col in df_h3.columns:
            cell_metadata[col] = df_h3.groupby(h3_column)[col].apply(safe_mode).values

    df_agg = df_agg.merge(cell_metadata, on=h3_column, how='left', suffixes=('', '_celula'))

    df_agg['latitude'] = df_agg['latitude'].fillna(df_agg['latitude_celula'])
    df_agg['longitude'] = df_agg['longitude'].fillna(df_agg['longitude_celula'])
    for col in categorical_columns:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].fillna(df_agg[f"{col}_celula"])

    df_agg.drop(columns=[c for c in df_agg.columns if '_celula' in c], inplace=True)

    fill_zero_cols = vehicle_columns + victim_columns + ['holiday', 'weekend']
    for col in fill_zero_cols:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].fillna(0)

    print(f"\nAggregation complete. Total records: {len(df_agg):,}")
    return df_agg

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sinusoidal and cosinoidal columns for cyclic time variables.

    Expects the DataFrame to have columns:
    'day_of_week', 'month', 'day_of_year', 'hour' (optional)
    """
    df = df.copy()
    
    if 'day_of_week' in df:
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    if 'month' in df:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'day_of_year' in df:
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    if 'hour' in df:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    if 'week_of_year' in df:
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
    
    return df
