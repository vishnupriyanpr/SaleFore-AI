import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from config import *
from utils import *

def load_data():
    try:
        df = pd.read_parquet(PROCESSED_TRAIN)
    except Exception:
        df = pd.read_csv(str(PROCESSED_TRAIN).replace('.parquet', '.csv'))
    return df

def preprocess(X, scaler=None, fit=False):
    Xproc = X.copy()
    for col in Xproc.columns:
        Xproc[col] = pd.to_numeric(Xproc[col], errors='coerce')
    Xproc = Xproc.fillna(0)
    if scaler is None:
        scaler = RobustScaler()
        Xproc = scaler.fit_transform(Xproc)
        return pd.DataFrame(Xproc, columns=X.columns), scaler
    else:
        Xproc = scaler.transform(Xproc)
        return pd.DataFrame(Xproc, columns=X.columns)

def main():
    print("Loading training data...")
    df = load_data()
    features = [col for col in df.columns if col not in ['Date', 'Sales', 'Store', 'Customers']]
    X = df[features]
    y = df['Sales']

    # Simple time split: last 10% as validation.
    split_idx = int(0.9 * len(df))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scaling
    X_train_proc, scaler = preprocess(X_train, fit=True)
    X_val_proc = preprocess(X_val, scaler=scaler, fit=False)
    joblib.dump(scaler, MODELS_DIR / "scalers.joblib")

    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=8, random_state=RANDOM_STATE, tree_method='auto', verbosity=0)
    xgb_model.fit(X_train_proc, y_train)
    joblib.dump(xgb_model, XGBOOST_MODEL)

    # LightGBM
    print("Training LightGBM...")
    lgbm_model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, random_state=RANDOM_STATE)
    lgbm_model.fit(X_train_proc, y_train)
    joblib.dump(lgbm_model, LIGHTGBM_MODEL)

    print("Evaluating models on validation data:")
    for name, model in [('XGBoost', xgb_model), ('LightGBM', lgbm_model)]:
        preds = model.predict(X_val_proc)
        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds, squared=False)
        r2 = r2_score(y_val, preds)
        print(f"{name:8s}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")

    print("\nModels saved & ready for forecasting!")

if __name__ == "__main__":
    main()
