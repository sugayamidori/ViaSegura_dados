import pandas as pd
import numpy as np
import holidays
from datetime import date, timedelta
from dateutil.easter import easter
from src.utils import add_cyclic_features


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
    print("CREATING TEMPORAL FEATURES")
    
    df = df.copy()
    
    # Basic temporal components
    df['year'] = df['Data'].dt.year
    df['month'] = df['Data'].dt.month
    df['day'] = df['Data'].dt.day
    df['day_of_week'] = df['Data'].dt.dayofweek
    df['day_of_year'] = df['Data'].dt.dayofyear
    df['week_of_year'] = df['Data'].dt.isocalendar().week
    df['quarter'] = df['Data'].dt.quarter
    
    # Cyclic features
    df = add_cyclic_features(df)
    
    # Time periods
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    for period_name, (start, end) in time_periods.items():
        df[period_name] = ((df['hour'] >= start) & (df['hour'] <= end)).astype(int)
    
    # Holidays
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
    
    df['holiday'] = df['Data'].apply(lambda x: 1 if x in br_holidays else 0)
    
    return df