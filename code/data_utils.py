import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file_path, target_column='PCE', log_transform=True):
    """
    从指定路径加载 CSV 数据，进行初步清洗，并分离特征和目标变量。
    
    参数:
        file_path (str): CSV 文件的完整路径。
        target_column (str): 目标变量的列名，默认为 'PCE'。
        log_transform (bool): 是否对目标变量进行 log1p 变换。
        
    返回:
        X (pd.DataFrame): 特征数据。
        y (pd.Series): 目标变量数据。
    """
    print(f"\n🚀 正在从 {file_path} 读取数据...")
    df = pd.read_csv(file_path)
    
    # 如果存在 'Unnamed: 0' 列，则删除它
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("已删除 'Unnamed: 0' 列。")

    # 检查目标列是否存在
    if target_column not in df.columns:
        raise ValueError(f"❌ 错误: 目标列 '{target_column}' 在数据集中不存在。")
        
    # 删除标签列为空的行
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    if len(df) < initial_rows:
        print(f"警告: 已删除 {initial_rows - len(df)} 行，因为目标列 '{target_column}' 存在缺失值。")

    # 提取目标变量 y 和特征 X
    y = df.pop(target_column)
    X = df
    
    if log_transform:
        y = np.log1p(y)
        print(f"✅ 已对目标变量 '{target_column}' 进行 log1p 变换。")
        
    print(f"✅ 数据加载完成。特征形状: {X.shape}, 目标形状: {y.shape}")
    return X, y

def clean_data_manual(
    X,
    y,
    variance_threshold=0.0,
    missing_ratio_limit=0.3,
    correlation_threshold=0.95
    ):
    """
    对特征数据进行清洗，包括处理错误码、缺失值过滤、方差过滤、相关性过滤和标准化。
    保持特征与标签的索引对齐。
    """
    print(f"\n开始数据清洗流程...")
    
    if isinstance(y, pd.Series):
        y_df = y.to_frame()
    else:
        y_df = y
        
    label_col = y_df.columns[0]

    # 1. 同步准备：强制合并特征和标签以确保对齐
    df_combined = pd.concat([X, y_df], axis=1, join='inner')

    # 2. 行级别清洗
    # 2.1 删除标签缺失的行 (虽然 load_data 已处理，此处作为双重保险)
    df_combined = df_combined.dropna(subset=[label_col])

    # 2.2 处理 Mordred 错误码 (将非数值转为 NaN)
    X_temp = df_combined.drop(columns=[label_col])
    y_temp = df_combined[label_col]
    X_temp = X_temp.apply(pd.to_numeric, errors='coerce')

    # 2.3 删除完全重复的样本
    df_combined = pd.concat([X_temp, y_temp], axis=1).drop_duplicates()
    
    # 3. 特征级别清洗
    X_curr = df_combined.drop(columns=[label_col])
    y_curr = df_combined[label_col]

    # 3.1 缺失率过滤
    missing_ratios = X_curr.isnull().mean()
    cols_to_keep = missing_ratios[missing_ratios <= missing_ratio_limit].index
    X_curr = X_curr[cols_to_keep]
    print(f"删除缺失率 > {missing_ratio_limit*100}% 的特征后维度: {X_curr.shape}")

    # 3.2 方差过滤
    vars_series = X_curr.var()
    cols_var = vars_series[vars_series > variance_threshold].index
    X_curr = X_curr[cols_var]
    print(f"删除方差 <= {variance_threshold} 的常量特征后维度: {X_curr.shape}")

    # 3.3 高相关性过滤
    if correlation_threshold and correlation_threshold < 1.0:
        corr_matrix = X_curr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        X_curr = X_curr.drop(columns=to_drop)
        print(f"删除相关性 > {correlation_threshold} 的高度相关特征后维度: {X_curr.shape}")

    # 4. 缺失值填充 (中位数)
    imputer = SimpleImputer(strategy='median')
    X_imputed_val = imputer.fit_transform(X_curr)
    X_imputed = pd.DataFrame(X_imputed_val, columns=X_curr.columns, index=X_curr.index)

    # 5. 特征缩放 (标准化)
    scaler = StandardScaler()
    X_scaled_val = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled_val, columns=X_imputed.columns, index=y_curr.index)

    print(f"✅ 数据清洗与标准化完成。最终特征维度: {X_scaled.shape}")
    return X_scaled, y_curr

def auto_pca_reduction(df_scaled, target_variance=0.90, verbose=True):
    """
    自动寻找最小主成分数量以满足目标累计解释方差比例，并进行 PCA 降维。
    """
    pca_full = PCA()
    pca_full.fit(df_scaled)
    cum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    best_n = np.argmax(cum_variance >= target_variance) + 1

    if verbose:
        print(f"\n📊 累计方差分析:")
        print(f"   - 解释 {target_variance*100}% 的方差，需要前 {best_n} 个主成分。")
        print(f"   - 维度压缩率: {(1 - best_n/df_scaled.shape[1])*100:.2f}%")

    pca_final = PCA(n_components=best_n)
    data_pca = pca_final.fit_transform(df_scaled)
    
    column_names = [f'PC{i+1}' for i in range(best_n)]
    df_pca = pd.DataFrame(data_pca, columns=column_names, index=df_scaled.index)

    return df_pca, pca_final
