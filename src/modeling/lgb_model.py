# lgb_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.config.config import USE_GPU, GPU_DEVICE_ID

def poisson_deviance(y_true, y_pred):
    """Calculates Poisson Deviance (lower is better)."""
    y_pred = np.clip(y_pred, 1e-9, None)  # Evita log(0)
    # y_true == 0, 0 * log(0 / y_pred) = 0
    term1 = np.where(y_true == 0, 0, y_true * np.log(y_true / y_pred))
    return 2 * np.mean(term1 - (y_true - y_pred))

def train_lgb_model(X, y, feature_cols):
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    params = {
        'objective': 'poisson',
        'metric': 'poisson',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }
    
    if USE_GPU:
        params['device'] = 'gpu'
        params['gpu_device_id'] = GPU_DEVICE_ID
        print("LightGBM set for GPU")
    else:
        params['device'] = 'cpu'
        params['num_threads'] = -1  
        print("LightGBM set for CPU")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold}/5...", end=" ", flush=True)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=200)

        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        deviance = poisson_deviance(y_test, y_pred)
        
        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, Poisson Deviance={deviance:.4f}", flush=True)
        
        results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'poisson_deviance': deviance
        })

    return results