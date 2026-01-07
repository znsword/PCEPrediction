import os

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Mordred_descriptors_data_PCE_revised.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- 数据处理参数 ---
TARGET_COLUMN = 'PCE'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 数据清洗参数
CLEANING_PARAMS = {
    'variance_threshold': 0.0,
    'missing_ratio_limit': 0.3,
    'correlation_threshold': 0.95
}

# PCA 参数
PCA_TARGET_VARIANCE = 0.90

# --- MLP 模型默认超参数 ---
DEFAULT_MODEL_PARAMS = {
    'hidden_dim': 64,
    'output_dim': 1,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'batch_size': 32
}

# --- 超参数搜索空间 (Tuning) ---
TUNING_SEARCH_SPACE = {
    'hidden_dims': [32, 64, 128],
    'learning_rates': [0.01, 0.001, 0.0001],
    'num_epochs_list': [50, 100, 200],
    'batch_sizes': [16, 32],
    'n_splits': 5
}
