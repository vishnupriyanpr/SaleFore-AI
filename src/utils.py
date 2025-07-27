import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, filepath, model_type='joblib'):
    """Save model to specified filepath"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'joblib':
        joblib.dump(model, filepath)
    elif model_type == 'xgboost':
        model.save_model(str(filepath))
    elif model_type == 'lightgbm':
        model.booster_.save_model(str(filepath))
    print(f"Model saved to {filepath}")

def load_model(filepath, model_type='joblib'):
    """Load model from filepath"""
    if model_type == 'joblib':
        return joblib.load(filepath)
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(filepath))
        return model
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(filepath))

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def plot_forecast(actual, forecast, title="Sales Forecast", save_path=None):
    """Plot actual vs forecast"""
    plt.figure(figsize=(15, 8))
    plt.plot(actual.index, actual.values, label='Actual', linewidth=2)
    plt.plot(forecast.index, forecast.values, label='Forecast', linewidth=2, linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
