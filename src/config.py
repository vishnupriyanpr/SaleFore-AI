import os
from pathlib import Path
import torch

# Check GPU availability with PyTorch 2.10
GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = "cuda:0" if GPU_AVAILABLE else "cpu"
CUDA_VERSION = torch.version.cuda

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {CUDA_VERSION}")
print(f"GPU Available: {GPU_AVAILABLE}")

if GPU_AVAILABLE:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"GPU Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Set memory management for PyTorch 2.10
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FORECASTS_DIR = PROJECT_ROOT / "data" / "forecasts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
STORE_FILE = DATA_RAW_DIR / "store.csv"
TRAIN_FILE = DATA_RAW_DIR / "train.csv" 
TEST_FILE = DATA_RAW_DIR / "test.csv"
PROCESSED_TRAIN = DATA_PROCESSED_DIR / "train_processed.parquet"
FORECAST_CSV = FORECASTS_DIR / "sales_forecast_results.csv"

# Model files
XGBOOST_MODEL = MODELS_DIR / "xgboost_model.json"
LIGHTGBM_MODEL = MODELS_DIR / "lightgbm_model.txt"
RF_MODEL = MODELS_DIR / "rf_model.joblib"

# RTX 4060 optimized parameters (8GB VRAM)
PRED_HORIZON = 6
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTUNA_TRIALS = 50  # Optimized for RTX 4060
CV_FOLDS = 3
GPU_MEMORY_FRACTION = 0.75  # Conservative for 8GB VRAM

# Create directories
for directory in [DATA_PROCESSED_DIR, FORECASTS_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
