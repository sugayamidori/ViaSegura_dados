import pandas as pd
from src.modeling.weekly_aggregration import aggregate_weekly
from src.modeling.lgb_model import train_and_export_model

def main():
    print("VIASEGURA PIPELINE START")
    print("=" * 40)

    df_weekly = aggregate_weekly()  

    if not pd.api.types.is_datetime64_any_dtype(df_weekly['week_start']):
        df_weekly['week_start'] = pd.to_datetime(df_weekly['week_start'])

    feature_cols = [
        'latitude', 'longitude',
        'year', 'month',
        'week_sin', 'week_cos',
        'month_sin', 'month_cos'
    ]

    missing = [col for col in feature_cols if col not in df_weekly.columns]
    if missing:
        raise ValueError(f"Features ausentes no dataset: {missing}")

    model, results = train_and_export_model(df_weekly, feature_cols)

    print(" MODEL TRAINED AND EXPORTED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()