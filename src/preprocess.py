import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import *

class SalesDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self):
        """Load and merge datasets"""
        print("Loading datasets...")
        
        # Load store data
        store_df = pd.read_csv(STORE_FILE)
        print(f"Store data shape: {store_df.shape}")
        
        # Load train data  
        train_df = pd.read_csv(TRAIN_FILE)
        print(f"Train data shape: {train_df.shape}")
        
        # Load test data if available
        test_df = None
        if TEST_FILE.exists():
            test_df = pd.read_csv(TEST_FILE)
            print(f"Test data shape: {test_df.shape}")
        
        return store_df, train_df, test_df
    
    def fix_data_types(self, df):
        """Fix data type issues for Parquet compatibility"""
        # Handle StateHoliday column specifically
        if 'StateHoliday' in df.columns:
            # Convert all values to string first, then encode
            df['StateHoliday'] = df['StateHoliday'].astype(str)
        
        # Handle SchoolHoliday if it exists
        if 'SchoolHoliday' in df.columns:
            df['SchoolHoliday'] = df['SchoolHoliday'].astype(int)
        
        # Ensure all object columns are properly handled
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col not in ['Date']:  # Don't convert Date column
                df[col] = df[col].astype(str)
        
        # Ensure numeric columns are proper numeric types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def advanced_feature_engineering(self, train_df, store_df):
        """Advanced feature engineering for sales forecasting"""
        print("Starting feature engineering...")
        
        # Fix data types before merging
        train_df = self.fix_data_types(train_df)
        store_df = self.fix_data_types(store_df)
        
        # Merge datasets
        df = train_df.merge(store_df, on='Store', how='left')
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        # Remove closed stores (Sales = 0)
        df = df[df['Open'] == 1].copy()
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        # Handle StateHoliday properly
        if 'StateHoliday' in df.columns:
            # Create binary holiday indicator
            df['IsStateHoliday'] = (df['StateHoliday'] != '0').astype(int)
            # Encode the actual holiday types
            le_state = LabelEncoder()
            df['StateHoliday_Encoded'] = le_state.fit_transform(df['StateHoliday'])
            self.label_encoders['StateHoliday'] = le_state
        
        # Competition features
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
        
        # Calculate competition open duration
        df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                               (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
        
        # Promo2 features
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0) 
        df['PromoInterval'] = df['PromoInterval'].fillna('None')
        
        # Create Promo2 active indicator
        def is_promo2_active(row):
            if row['Promo2SinceWeek'] == 0:
                return 0
            promo_months = {
                'Jan,Apr,Jul,Oct': [1,4,7,10], 
                'Feb,May,Aug,Nov': [2,5,8,11], 
                'Mar,Jun,Sept,Dec': [3,6,9,12]
            }
            if row['PromoInterval'] in promo_months:
                return 1 if row['Month'] in promo_months[row['PromoInterval']] else 0
            return 0
        
        df['Promo2Active'] = df.apply(is_promo2_active, axis=1)
        
        # Lag features
        print("Creating lag features...")
        for lag in [1, 7, 14, 30, 60]:
            df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
            df[f'Customers_Lag_{lag}'] = df.groupby('Store')['Customers'].shift(lag)
        
        # Rolling statistics
        print("Creating rolling features...")
        for window in [7, 14, 30, 60]:
            df[f'Sales_Rolling_Mean_{window}'] = df.groupby('Store')['Sales'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'Sales_Rolling_Std_{window}'] = df.groupby('Store')['Sales'].rolling(window=window).std().reset_index(0, drop=True)
            df[f'Customers_Rolling_Mean_{window}'] = df.groupby('Store')['Customers'].rolling(window=window).mean().reset_index(0, drop=True)
        
        # Exponential weighted features
        for alpha in [0.1, 0.3, 0.5]:
            df[f'Sales_EWM_{alpha}'] = df.groupby('Store')['Sales'].ewm(alpha=alpha).mean().reset_index(0, drop=True)
        
        # Store-specific statistics
        store_stats = df.groupby('Store')['Sales'].agg(['mean', 'std', 'median']).reset_index()
        store_stats.columns = ['Store', 'Store_Sales_Mean', 'Store_Sales_Std', 'Store_Sales_Median']
        df = df.merge(store_stats, on='Store', how='left')
        
        # Encode categorical variables
        categorical_cols = ['StoreType', 'Assortment', 'PromoInterval']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Fill remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Final data type fixes for Parquet
        df = self.fix_data_types(df)
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df

def main():
    """Main preprocessing pipeline"""
    preprocessor = SalesDataPreprocessor()
    
    # Load data
    store_df, train_df, test_df = preprocessor.load_data()
    
    # Process training data
    processed_df = preprocessor.advanced_feature_engineering(train_df, store_df)
    
    # Alternative: Save as CSV if Parquet continues to have issues
    try:
        processed_df.to_parquet(PROCESSED_TRAIN, index=False)
        print(f"Processed training data saved to {PROCESSED_TRAIN}")
    except Exception as e:
        print(f"Parquet save failed: {e}")
        print("Saving as CSV instead...")
        csv_path = PROCESSED_TRAIN.with_suffix('.csv')
        processed_df.to_csv(csv_path, index=False)
        print(f"Processed training data saved to {csv_path}")
        
        # Update config to use CSV
        import config
        config.PROCESSED_TRAIN = csv_path
    
    # Save label encoders
    joblib.dump(preprocessor.label_encoders, MODELS_DIR / "label_encoders.joblib")
    print("Label encoders saved")

if __name__ == "__main__":
    main()
