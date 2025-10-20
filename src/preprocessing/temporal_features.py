"""
Temporal feature engineering.
"""

import pandas as pd
import numpy as np
import holidays
from datetime import date, timedelta
from dateutil.easter import easter


def create_temporal_features(
    df: pd.DataFrame,
    time_periods: dict,
    municipal_holidays: dict
) -> pd.DataFrame:
    """
    Create all temporal features.
    
    Args:
        df: DataFrame with 'DATA' column
        time_periods: Dict of time period definitions
        municipal_holidays: Dict of municipal holidays
    
    Returns:
        DataFrame with temporal features
    """
    print("\n" + "="*70)
    print("CREATING TEMPORAL FEATURES")
    print("="*70)
    
    df = df.copy()
    
    # Basic temporal components
    print("\n[1/4] Extracting basic temporal components...")
    df['year'] = df['DATA'].dt.year
    df['month'] = df['DATA'].dt.month
    df['day'] = df['DATA'].dt.day
    df['day_of_week'] = df['DATA'].dt.dayofweek
    df['day_of_year'] = df['DATA'].dt.dayofyear
    df['week_of_year'] = df['DATA'].dt.isocalendar().week
    df['quarter'] = df['DATA'].dt.quarter
    
    # Cyclic features
    print("\n[2/4] Creating cyclic features...")
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Time periods
    print("\n[3/4] Creating time period features...")
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    for period_name, (start, end) in time_periods.items():
        df[period_name] = ((df['hour'] >= start) & (df['hour'] <= end)).astype(int)
    
    # Holidays
    print("\n[4/4] Adding holiday features...")
    years = sorted(df['year'].unique().astype(int))
    br_holidays = holidays.Brazil(years=years)
    
    # Municipal holidays
    for year in years:
        for name, (month, day) in municipal_holidays.items():
            br_holidays[date(year, month, day)] = name
    
    # Moveable holidays
    for year in years:
        easter_date = easter(year)
        carnival = easter_date - timedelta(days=47)
        good_friday = easter_date - timedelta(days=2)
        br_holidays[carnival] = "Carnaval"
        br_holidays[good_friday] = "Good Friday"
    
    df['holiday'] = df['DATA'].apply(lambda x: 1 if x in br_holidays else 0)
    
    print(f"\n✓ Temporal features created")
    print(f"  - Cyclic: dow_sin/cos, month_sin/cos, doy_sin/cos, hour_sin/cos")
    print(f"  - Binary: weekend, {', '.join(time_periods.keys())}, holiday")
    print("="*70)
    
    return df