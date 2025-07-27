import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from config import *
from utils import *

def load_models():
    models = {}
    models['xgboost'] = joblib.load(XGBOOST_MODEL)
    models['lightgbm'] = joblib.load(LIGHTGBM_MODEL)
    return models

def load_scaler():
    try:
        return joblib.load(MODELS_DIR / "scalers.joblib")
    except Exception:
        return None

def preprocess_features(df, feature_cols, scaler):
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X

def create_forecast_horizon(df, periods=6, freq='MS', date_col='Date'):
    last_date = pd.to_datetime(df[date_col]).max()
    horizon_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq=freq)
    template = df.iloc[[-1]].copy()
    future_df = pd.concat([template]*periods, ignore_index=True)
    future_df[date_col] = horizon_dates
    for col in future_df.columns:
        future_df[col] = pd.to_numeric(future_df[col], errors='coerce')
    return pd.concat([df, future_df], ignore_index=True), future_df

def main():
    # Load training data
    if os.path.exists(PROCESSED_TRAIN):
        try:
            df = pd.read_parquet(PROCESSED_TRAIN)
        except Exception:
            df = pd.read_csv(str(PROCESSED_TRAIN).replace('.parquet', '.csv'))
    else:
        print("Processed train file not found."); sys.exit(1)

    periods = PRED_HORIZON if 'PRED_HORIZON' in globals() else 6
    full_df, future_df = create_forecast_horizon(df, periods=periods, freq='MS')
    feature_cols = [col for col in df.columns if col not in ['Date', 'Sales', 'Store', 'Customers']]

    models = load_models()
    scaler = load_scaler()
    X_future = preprocess_features(future_df, feature_cols, scaler)

    # Get predictions for each model
    preds = {}
    for name, model in models.items():
        try:
            preds[name] = model.predict(X_future)
        except Exception as e:
            print(f"{name} prediction failed: {e}")
            preds[name] = np.full(X_future.shape[0], np.nan)

    # Equal-weight ensemble
    weights = {k: 1/len(preds) for k in preds}
    ensemble_pred = sum(weights[k]*preds[k] for k in preds)
    for name in preds:
        future_df[f"{name}_forecast"] = preds[name]
    future_df['ML_Ensemble_Pred'] = ensemble_pred

    # Write Power BI-ready CSV
    out_cols = ['Date', 'ML_Ensemble_Pred'] + [f"{k}_forecast" for k in preds]
    out_path = FORECASTS_DIR / 'sales_forecast_results.csv'
    future_df[out_cols].to_csv(out_path, index=False)
    print(f"Forecast saved to: {out_path}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(future_df['Date'], future_df['ML_Ensemble_Pred'], marker='o', label='ML Ensemble')
    for k in preds:
        plt.plot(future_df['Date'], future_df[f"{k}_forecast"], marker='.', linestyle='--', label=f"{k} Forecast")
    plt.title('Future Sales Forecast')
    plt.xlabel('Date'); plt.ylabel('Forecasted Sales')
    plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
