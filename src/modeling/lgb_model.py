import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_lgb_model(X, y, metadata, feature_cols):
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        params = {
            'objective': 'poisson',
            'metric': 'poisson',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=200)

        y_pred = model.predict(X_test)
        results.append({
            'fold': fold,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        })

    return results
