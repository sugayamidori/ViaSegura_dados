
import pandas as pd
import numpy as np
import os

from src.config.config import GEOCODE_FULL_COMPLETE, RECIFE_H3_GRID_PATH, WEEKLY_DATASET_PATH

def aggregate_weekly():
    """
    Aggregate accidents per week and hexagon (H3).
    Output: final dataset for modeling.
    """
    print("="*60)
    print("WEEKLY AGGREGATION")
    print("="*60)
    
    #Load pre-processed data 
    df = pd.read_csv(GEOCODE_FULL_COMPLETE)
    
    # Validate datetime 
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    invalid_dates = df['Data'].isna().sum()
    if invalid_dates > 0:
        print(f" Removed {invalid_dates:,} invalid dates")
        df = df.dropna(subset=['Data'])
    
    print(f"Period: {df['Data'].min().date()} → {df['Data'].max().date()}")
    
    #  Verify h3_cell exists
    if 'h3_cell' not in df.columns:
        raise ValueError(" h3_cell column not found! geocoded_full.csv must have H3 cells.")
    
    before = len(df)
    df = df[df['h3_cell'].notna()]
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed:,} records without h3_cell")
    
    print(f" H3 cells: {df['h3_cell'].nunique():,} unique")
    
    # Create week_start (Monday of the week)
    print("\n Creating week_start")
    df['week_start'] = df['Data'].dt.to_period('W').dt.start_time
    print(f" Weeks: {df['week_start'].nunique()} unique")
    
    # Aggregate by h3_cell and week_start)
    print("\n Aggregating by (hex × week)")
    agg_base = df.groupby(['h3_cell', 'week_start']).size().reset_index(name='num_sinistros')
    print(f"Combinations with accidents: {len(agg_base):,}")
    
    # Full grid (all hexagons × all weeks)
    print("\n Creating full grid (zero-filling)")
    
    # Load Recife grid
    grid = pd.read_csv(RECIFE_H3_GRID_PATH)
    print(f" Recife grid: {len(grid):,} hexágonos")

    col_hex = None
    for c in ['hex_id', 'h3_address', 'id', 'h3_cell']:
        if c in grid.columns:
            col_hex = c
            break
            
    if not col_hex:
        col_hex = grid.columns[0]
        
    print(f"   Grid loaded ({len(grid):,} hexs). Using column: '{col_hex}'")
    
    # All weeks
    all_weeks = agg_base['week_start'].drop_duplicates().sort_values()
    print(f"Total weeks: {len(all_weeks)}")
    
    from itertools import product
    full_grid = pd.DataFrame(
        list(product(grid['hex_id'], all_weeks)),
        columns=['h3_cell', 'week_start']
    )
    print(f" Total combinations: {len(full_grid):,}")
    
    #. left join with actual data ===
    print("\n Merging with actual accident data")
    df_weekly = full_grid.merge(agg_base, on=['h3_cell', 'week_start'], how='left')
    
    df_weekly['num_sinistros'] = df_weekly['num_sinistros'].fillna(0).astype(int)
    
    # Statistics
    zeros = (df_weekly['num_sinistros'] == 0).sum()
    print(f" Merged: {len(df_weekly):,} total records")
    print(f" Zeros: {zeros:,} ({zeros/len(df_weekly)*100:.1f}%)")
    print(f" Non-zeros: {len(df_weekly) - zeros:,} ({(1-zeros/len(df_weekly))*100:.1f}%)")
    
    # Add spatial metadata (centroid of each hex) ===
    print("\nAdding spatial metadata")
    cell_meta = df.groupby('h3_cell').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'bairro_clean': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
    }).reset_index()
    
    df_weekly = df_weekly.merge(cell_meta, on='h3_cell', how='left')
    print(f" Added lat/lon/bairro for {cell_meta['h3_cell'].nunique():,} hexágonos")
    
    print("\n Adding temporal features")
    df_weekly['year'] = df_weekly['week_start'].dt.year
    df_weekly['month'] = df_weekly['week_start'].dt.month
    df_weekly['week_of_year'] = df_weekly['week_start'].dt.isocalendar().week
    

    print("\n Adding cyclic features")
    df_weekly['week_sin'] = np.sin(2 * np.pi * df_weekly['week_of_year'] / 52.0)
    df_weekly['week_cos'] = np.cos(2 * np.pi * df_weekly['week_of_year'] / 52.0)
    df_weekly['month_sin'] = np.sin(2 * np.pi * df_weekly['month'] / 12)
    df_weekly['month_cos'] = np.cos(2 * np.pi * df_weekly['month'] / 12)
    
    print("\nSaving weekly dataset")
    df_weekly.to_csv(WEEKLY_DATASET_PATH, index=False)
    
    # Final statistics
    print("\n" + "="*60)
    print("WEEKLY AGGREGATION COMPLETE")
    print("="*60)
    print(f"\n OUTPUT:")
    print(f"   File: {WEEKLY_DATASET_PATH}")
    print(f"   Records: {len(df_weekly):,}")
    print(f"   Hexágonos: {df_weekly['h3_cell'].nunique():,}")
    print(f"   Weeks: {df_weekly['week_start'].nunique()}")
    print(f"   Period: {df_weekly['week_start'].min().date()} → {df_weekly['week_start'].max().date()}")
    
    print(f"\n TARGET (num_sinistros):")
    print(f"   Mean: {df_weekly['num_sinistros'].mean():.4f}")
    print(f"   Std: {df_weekly['num_sinistros'].std():.4f}")
    print(f"   Var: {df_weekly['num_sinistros'].var():.4f}")
    print(f"   Min: {df_weekly['num_sinistros'].min()}")
    print(f"   Max: {df_weekly['num_sinistros'].max()}")
    print(f"   Dispersion index (var/mean): {df_weekly['num_sinistros'].var() / df_weekly['num_sinistros'].mean():.4f}")
    
    print(f"\n COLUMNS ({len(df_weekly.columns)}):")
    print(f"   {list(df_weekly.columns)}")
    
    print("="*60 + "\n")
    
    return df_weekly

if __name__ == "__main__":
    aggregate_weekly()