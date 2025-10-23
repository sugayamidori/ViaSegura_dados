import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from utils import add_cyclic_features

def train_poisson(df):
    agg_df = df.groupby(['h3_cell', 'Data']).agg(
        sinistros=('h3_cell', 'count'),
        year=('year', 'first'),
        month=('month', 'first'),
        day_of_week=('day_of_week', 'first'),
        holiday=('holiday', 'max')
    ).reset_index()

    agg_df = add_cyclic_features(agg_df)

    X = agg_df[['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'holiday']]
    y = agg_df['sinistros']

    model = PoissonRegressor(alpha=1e-6, max_iter=1000)
    model.fit(X, y)

    y_pred = model.predict(X)
    return {
        'model': model,
        'metrics': {
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'Poisson Deviance': mean_poisson_deviance(y, np.clip(y_pred, 1e-9, None))
        }
    }
