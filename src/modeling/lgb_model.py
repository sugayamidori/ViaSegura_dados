import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import json
from src.config.config import USE_GPU, GPU_DEVICE_ID, WEEKLY_DATASET_PATH, BACKEND_EXPORT_DIR,N_BOOST_ROUNDS, N_SPLITS, N_BOOST_ROUNDS
import time 

def poisson_deviance(y_true, y_pred):
    """Calculates Poisson Deviance (lower is better)."""
    y_pred = np.clip(y_pred, 1e-9, None)
    term1 = np.where(y_true == 0, 0, y_true * np.log(y_true / y_pred))
    return 2 * np.mean(term1 - (y_true - y_pred))

def train_and_export_model(df_weekly, feature_cols, target_col='num_sinistros'):
    """
    TRAINING MODEL WITH GPU AND TIME SERIES SPLIT
    """
    
    X = df_weekly[feature_cols].copy()
    y = df_weekly[target_col].copy()

    tscv = TimeSeriesSplit(N_SPLITS)
    results = []

    params = {
        'objective': 'regression_l2', 
        'metric': ['mae', 'rmse'],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'min_data_in_leaf': 20,
        'verbose': -1
    }

    if USE_GPU:
        params['device'] = 'gpu'
        params['gpu_device_id'] = GPU_DEVICE_ID
        print(" LightGBM set for GPU")
    else:
        params['device'] = 'cpu'
        params['num_threads'] = -1
        print(" LightGBM set for CPU")

    final_model = None

    fold_times = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold}/5...", end=" ", flush=True)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        
        # CPU x GPU
        start_time = time.time()
        model = lgb.train(params, train_data, N_BOOST_ROUNDS)
        fold_time = time.time() - start_time
        fold_times.append(fold_time)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        deviance = poisson_deviance(y_test, y_pred)

        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, Poisson Deviance={deviance:.4f} [{fold_time:.2f}s]", flush=True)

        results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'poisson_deviance': deviance
        })

        if fold == tscv.n_splits:
            final_model = model

    print(" Generating predictions for full dataset")
    df_weekly['predicted_num_sinistros'] = final_model.predict(X)

    avg_time = np.mean(fold_times)
    total_time = np.sum(fold_times)
    device = "GPU" if USE_GPU else "CPU"
    print(f"\n Training finished in {total_time:.2f}s ({avg_time:.2f}s/fold) with {device}")
    

    predictions_path = os.path.join(BACKEND_EXPORT_DIR, "predictions_weekly.csv")
    df_weekly[['h3_cell', 'week_start', 'predicted_num_sinistros']].to_csv(predictions_path, index=False)
    print(f" Predictions saved to: {predictions_path}")

    #  heatmap_monthly.csv (aggregated month x hex)
    print("Generating monthly heatmap")
    df_monthly = df_weekly.copy()
    df_monthly['month_year'] = df_monthly['week_start'].dt.to_period('M').dt.start_time
    heatmap = df_monthly.groupby(['h3_cell', 'month_year'])['predicted_num_sinistros'].sum().reset_index()
    heatmap.rename(columns={'month_year': 'month_start'}, inplace=True)
    heatmap_path = os.path.join(BACKEND_EXPORT_DIR, "heatmap_monthly.csv")
    heatmap.to_csv(heatmap_path, index=False)
    print(f" Monthly heatmap saved to: {heatmap_path}")

    
    print("Exporting H3 grid with metadata")
    grid_df = pd.read_csv(WEEKLY_DATASET_PATH)[['h3_cell', 'latitude', 'longitude', 'bairro_clean']].drop_duplicates()
    grid_export_path = os.path.join(BACKEND_EXPORT_DIR, "h3_grid.csv")
    grid_df.to_csv(grid_export_path, index=False)
    print(f" H3 grid saved to: {grid_export_path}")

    
    metadata = {
        "model_type": "LightGBM",
        "objective": params['objective'],
        "features": feature_cols,
        "target": target_col,
        "n_folds": tscv.n_splits,
        "use_gpu": USE_GPU,
        "gpu_device_id": GPU_DEVICE_ID if USE_GPU else None,
        "results": results,
        "timestamp": pd.Timestamp.now().isoformat(), 
        "fold_times_seconds": fold_times, 
        "training_time_seconds": total_time,
        "avg_fold_time_seconds": avg_time,
        "device_used": device
    }
    meta_path = os.path.join(BACKEND_EXPORT_DIR, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f" Metadata saved to: {meta_path}")

    return final_model, results