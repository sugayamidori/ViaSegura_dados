import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

from src.utils import aggregate_weekly_by_h3
from src.config.config import (
    PROCESSED_DATASET_PATH,
    PANDEMIC_YEARS,
    FEATURE_COLUMNS,
    N_BOOST_ROUNDS,
    MODEL_CONFIG,
    USE_GPU,
    GPU_DEVICE_ID,
    BACKEND_EXPORT_DIR,
    PREDICTION_WEEKS
)
from src.modeling.lgb_model import train_lgb_model
from src.utils import add_cyclic_features

from src.modeling.poisson_model import train_poisson

def add_historical_features(df_weekly: pd.DataFrame) -> pd.DataFrame:
    df = df_weekly.sort_values(['h3_cell', 'week_start']).copy()
    df['week_start'] = pd.to_datetime(df['week_start'])
    grouped = df.groupby('h3_cell')

    df['sinistros_lag_1w'] = grouped['num_sinistros'].shift(1)
    df['sinistros_lag_4w'] = grouped['num_sinistros'].shift(4)
    df['sinistros_mean_4w'] = grouped['num_sinistros'].transform(
        lambda x: x.rolling(4, min_periods=1).mean().shift(1)
    )
    df['sinistros_mean_12w'] = grouped['num_sinistros'].transform(
        lambda x: x.rolling(12, min_periods=1).mean().shift(1)
    )
    df['total_historical_cell'] = grouped['num_sinistros'].cumsum().shift(1)

    for v in ['auto', 'moto', 'onibus', 'caminhao']:
        if v in df.columns:
            df[f'{v}_historical'] = grouped[v].cumsum().shift(1)

    hist_cols = [
        'sinistros_lag_1w', 'sinistros_lag_4w',
        'sinistros_mean_4w', 'sinistros_mean_12w',
        'total_historical_cell'
    ] + [f'{v}_historical' for v in ['auto', 'moto', 'onibus', 'caminhao'] if f'{v}_historical' in df.columns]
    df[hist_cols] = df[hist_cols].fillna(0)
    return df

def generate_predictions(model, df_historical, feature_cols, n_weeks=12):
    """
    Generates autoregressive forecasts for the next n_weeks.
    """
    last_week = df_historical['week_start'].max()
    h3_cells = df_historical['h3_cell'].unique()
    total_cells = len(h3_cells)

    print(f"  - Unique H3 cells: {total_cells:,}")
    print(f"  - Future weeks: {n_weeks}")
    print(f"  - Total predictions: {total_cells * n_weeks:,}")

    predictions = []
    df_future = df_historical.copy()

    for week_offset in range(1, n_weeks + 1):
        next_week_start = last_week + pd.Timedelta(weeks=week_offset)
        next_year = next_week_start.year
        next_week_num = next_week_start.isocalendar().week

        print(f"\n  [Week {week_offset}/{n_weeks}] Predicting for {next_week_start.date()}...")

        for idx, cell in enumerate(h3_cells):
            if idx % max(1, total_cells // 10) == 0:
                print(f"    → Progress: {idx}/{total_cells} cells ({100 * idx // total_cells}%)", end="\r")

            base_row = {
                'h3_cell': cell,
                'week_start': next_week_start,
                'year': next_year,
                'week_of_year': next_week_num,
                'month': next_week_start.month,
                'holiday': 0,
                'weekend': 1 if next_week_start.weekday() >= 5 else 0,
                'num_sinistros': 0
            }

            # Cyclic features
            base_row['month_sin'] = np.sin(2 * np.pi * base_row['month'] / 12)
            base_row['month_cos'] = np.cos(2 * np.pi * base_row['month'] / 12)
            base_row['week_sin'] = np.sin(2 * np.pi * base_row['week_of_year'] / 52.0)
            base_row['week_cos'] = np.cos(2 * np.pi * base_row['week_of_year'] / 52.0)

            # Historical context
            hist = df_future[df_future['h3_cell'] == cell].sort_values('week_start')
            if len(hist) > 0:
                base_row['sinistros_lag_1w'] = hist.iloc[-1]['num_sinistros']
                base_row['sinistros_lag_4w'] = hist['num_sinistros'].iloc[-4:].mean() if len(hist) >= 4 else 0
                base_row['sinistros_mean_4w'] = hist['num_sinistros'].iloc[-4:].mean()
                base_row['sinistros_mean_12w'] = hist['num_sinistros'].iloc[-12:].mean() if len(hist) >= 12 else hist['num_sinistros'].mean()
                base_row['total_historical_cell'] = hist['num_sinistros'].sum()

                for v in ['auto', 'moto', 'onibus', 'caminhao']:
                    col = f'{v}_historical'
                    base_row[col] = hist.iloc[-1][col] if col in hist.columns else 0
            else:
                for col in ['sinistros_lag_1w', 'sinistros_lag_4w', 'sinistros_mean_4w',
                            'sinistros_mean_12w', 'total_historical_cell']:
                    base_row[col] = 0
                for v in ['auto', 'moto', 'onibus', 'caminhao']:
                    base_row[f'{v}_historical'] = 0

            base_row['bairro_encoded'] = hist.iloc[-1]['bairro_encoded'] if 'bairro_encoded' in hist.columns and len(hist) > 0 else 0

            # Prediction
            X_row = pd.DataFrame([{col: base_row.get(col, 0) for col in feature_cols}])
            pred = model.predict(X_row)[0]
            base_row['predicted_accidents'] = max(0, pred)

            predictions.append({
                'h3_cell': cell,
                'week_start': next_week_start,
                'predicted_accidents': base_row['predicted_accidents']
            })

            base_row['num_sinistros'] = pred
            df_future = pd.concat([df_future, pd.DataFrame([base_row])], ignore_index=True)

        print(f"    → Progress: {total_cells}/{total_cells} cells (100%)")

    print(f"\n  Predictions generated: {len(predictions):,} records")
    return pd.DataFrame(predictions)

def export_backend_files(df_historical, df_predictions, model, feature_cols):
    export_dir = Path(BACKEND_EXPORT_DIR)
    export_dir.mkdir(exist_ok=True, parents=True)

    # 1. H3 grid metadata
    h3_meta = df_historical[['h3_cell', 'latitude', 'longitude', 'bairro_clean']].drop_duplicates()
    h3_meta.to_csv(export_dir / "h3_grid.csv", index=False)

    # 2. Weekly predictions
    df_predictions.to_csv(export_dir / "predictions_weekly.csv", index=False)

    # 3. Monthly heatmap (historical)
    df_historical['year_month'] = df_historical['week_start'].dt.to_period('M')
    monthly = df_historical.groupby(['h3_cell', 'year_month'])['num_sinistros'].sum().reset_index()
    monthly['year'] = monthly['year_month'].dt.year
    monthly['month'] = monthly['year_month'].dt.month
    monthly.drop(columns=['year_month'], inplace=True)
    monthly.to_csv(export_dir / "heatmap_monthly.csv", index=False)

    # 4. Metadata
    import json
    meta = {
        "last_updated": pd.Timestamp.now().isoformat(),
        "h3_resolution": 9,
        "prediction_weeks": PREDICTION_WEEKS,
        "model_type": "LightGBM Poisson",
        "total_h3_cells": len(h3_meta),
        "features_used": feature_cols
    }
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Exported backend files to: {export_dir}")


def main():
    print("VIASEGURA - FULL PIPELINE (TRAINING + EXPORT)")

    # 1. Load processed dataset
    print("\n[1/4] Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATASET_PATH, low_memory=False)

    print("\n[EXTRA] Training baseline Poisson model...")
    try:
        poisson_results = train_poisson(df.copy()) # Usa uma cópia para não alterar o df original
        print("  -> Poisson Model Metrics (on training data):")
        metrics = poisson_results['metrics']
        print(f"     - MAE: {metrics['MAE']:.4f}")
        print(f"     - RMSE: {metrics['RMSE']:.4f}")
        print(f"     - Poisson Deviance: {metrics['Poisson Deviance']:.4f}")
    except Exception as e:
        print(f"  -> Failed to train Poisson model. Error: {e}")

    # 2. Weekly aggregation
    print("\n[2/4] Aggregating by week and H3 cell...")
    df_weekly = aggregate_weekly_by_h3(
        df=df,
        pandemic_years=PANDEMIC_YEARS,
        vehicle_columns=['auto', 'moto', 'onibus', 'caminhao'],
        victim_columns=['vitimas', 'vitimasfatais'],
        categorical_columns=['bairro_clean']
    )

    # 3. Feature engineering
    df_features = add_historical_features(df_weekly)
    df_features = add_cyclic_features(df_features)

    # 4. Train model
    print("\n[3/4] Training LightGBM model...")
    available_features = [col for col in FEATURE_COLUMNS if col in df_features.columns]
    X = df_features[available_features]
    y = df_features['num_sinistros']

    params = MODEL_CONFIG.copy()
    if USE_GPU:
        params.update({'device': 'gpu', 'gpu_device_id': GPU_DEVICE_ID})
    else:
        params.update({'device': 'cpu', 'num_threads': -1})

    train_data = lgb.Dataset(X, label=y, feature_name=available_features)
    model = lgb.train(params, train_data, num_boost_round=N_BOOST_ROUNDS)
    results = train_lgb_model(X, y, available_features)  

    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_deviance = np.mean([r['poisson_deviance'] for r in results])

    print(f"\n✅ LightGBM CV Results → MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | Poisson Deviance: {avg_deviance:.4f}")

    # Save model
    export_dir = Path(BACKEND_EXPORT_DIR)
    export_dir.mkdir(exist_ok=True)
    model.save_model(export_dir / "lgb_model.txt")

    # 5. Predictions + export
    print("\n[4/4] Generating predictions and exporting for backend...")
    df_predictions = generate_predictions(model, df_features, available_features, PREDICTION_WEEKS)
    export_backend_files(df_features, df_predictions, model, available_features)

    print("\n🎉 PIPELINE SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
