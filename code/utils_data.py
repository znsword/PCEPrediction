import pandas as pd
import numpy as np

def load_pce_dataframe(csv_path, target_column='PCE', remove_unnamed=True, log_transform=True, verbose=True):
    df = pd.read_csv(csv_path)
    if remove_unnamed and 'Unnamed: 0' in df.columns:
        df = df.iloc[:, 1:]
        if verbose:
            print("Removed 'Unnamed: 0' column.")
    if target_column not in df.columns:
        if verbose:
            print(f"❌ 错误: 目标列 '{target_column}' 在数据集中不存在。")
        return None, None, df
    df = df.dropna(subset=[target_column])
    y = df.pop(target_column)
    if log_transform:
        y = np.log1p(y)
    X = df
    if verbose:
        print(f"Target variable y ('{target_column}') extracted. Shape: {y.shape}")
        print(f"Features X created. Shape: {X.shape}. Columns: {X.columns.tolist()[:5]}...")
    return X, y, df
