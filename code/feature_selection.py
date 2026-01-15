import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lgbm import LGBMRegressor # 确保已正确导入
from catboost import CatBoostRegressor

# ==========================================
# 1. 自定义转换器：高相关性过滤
# ==========================================
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    在 Pipeline 中移除彼此高度相关的特征。
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        # 将输入转为 DataFrame 以便计算相关性
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        # 选取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # 识别相关性大于阈值的列索引
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.drop(columns=self.to_drop).values

def train_and_evaluate_multi_model(
    X_train, X_test, y_train, y_test, 
    model_type='rf', 
    param_grid=None, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1, 
    show_plot=True, 
    verbose=True
):
    """
    通用多模型训练函数：集成一键切换 Yeo-Johnson 与 Quantile-Normal 变换。
    """
    
    # 1. 模型配置映射表
    model_map = {
        'rf': {
            'class': RandomForestRegressor,
            'fixed_params': {'random_state': 42},
            'grid': {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 10]
            }
        },
        'xgb': {
            'class': XGBRegressor,
            'fixed_params': {'random_state': 42, 'n_jobs': 1},
            'grid': {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [3, 6],
                'regressor__learning_rate': [0.1]
            }
        },
        'lgb': {
            'class': LGBMRegressor,
            'fixed_params': {'random_state': 42, 'n_jobs': 1, 'verbose': -1},
            'grid': {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.1]
            }
        },
        'cat': {
            'class': CatBoostRegressor,
            'fixed_params': {'random_state': 42, 'verbose': 0, 'allow_writing_files': False},
            'grid': {
                'regressor__iterations': [100, 200],
                'regressor__learning_rate': [0.1]
            }
        }
    }
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 2. 构建通用 Pipeline
    # 新增 'data_transform' 步骤作为占位符
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('data_transform', 'passthrough'), # <--- 变换一键切换的关键位置
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('correlation_filter', CorrelationFilter()), # 确保 CorrelationFilter 已定义
        ('scaler', StandardScaler()),
        ('regressor', model_map[model_type]['class'](**model_map[model_type]['fixed_params']))
    ])
    
    # 3. 确定参数网格
    if param_grid is None:
        param_grid = {
            # 这里实现一键对比三种数据处理方案
            'data_transform': [
                'passthrough',                                      # 方案 A: 不变换
                PowerTransformer(method='yeo-johnson'),            # 方案 B: Yeo-Johnson
                QuantileTransformer(output_distribution='normal')  # 方案 C: Quantile-Normal
            ],
            'correlation_filter__threshold': [0.95],
            **model_map[model_type]['grid']
        }
    
    # 4. 执行 GridSearchCV
    if verbose:
        print(f"开始 {model_type.upper()} 模型调优 (包含数据变换对比)...")
        
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    
    # 5. 提取结果与预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 6. 计算评估指标
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    
    # 7. 打印对比详情
    if verbose:
        # 从 cv_results_ 中提取变换方法的影响
        results_df = pd.DataFrame(grid_search.cv_results_)
        # 将对象转为字符串方便查看
        results_df['transform_method'] = results_df['param_data_transform'].apply(lambda x: str(x).split('(')[0])
        
        print("\n" + "="*50)
        print(f"最佳模型类型: {model_type.upper()}")
        print(f"最佳变换方案: {grid_search.best_params_['data_transform']}")
        print(f"最佳测试集 R²: {test_r2:.4f}")
        
        print("\n--- 不同变换方案的平均 CV R² 表现 ---")
        summary = results_df.groupby('transform_method')['mean_test_score'].max().sort_values(ascending=False)
        print(summary)
        print("="*50)
        
    # 可视化... (保持原代码不变)
    if show_plot:
        # ... (可视化逻辑)
        pass

    return {
        'model_type': model_type,
        'best_model': best_model,
        'best_transform': grid_search.best_params_['data_transform'],
        'test_r2': test_r2,
        'cv_results': grid_search.cv_results_
    }

    # ==========================================
# 5. 查看特征重要性与提取
# ==========================================
def analyze_feature_importance(
    pipeline, 
    original_feature_names, 
    top_n=50, 
    plot_top_n=20, 
    show_plot=True, 
    verbose=True
):
    """
    分析经过 Pipeline 处理后的特征重要性，并追踪特征名称的变化。
    """
    # 1. 提取关键步骤对象
    try:
        step_variance = pipeline.named_steps['variance_threshold']
        step_corr = pipeline.named_steps['correlation_filter']
        step_model = pipeline.named_steps['regressor']
    except KeyError as e:
        raise KeyError(f"Pipeline 中缺少必要的步骤名称: {e}。请检查 Pipeline 定义。")

    # 2. 追踪特征名称变化
    feature_names = list(original_feature_names)

    # A. 追踪 VarianceThreshold 的筛选结果
    vt_mask = step_variance.get_support()
    feature_names = [name for name, keep in zip(feature_names, vt_mask) if keep]
    n_after_variance = len(feature_names)
    
    if verbose:
        print(f"方差过滤后剩余特征数: {n_after_variance}")

    # B. 追踪 CorrelationFilter 的筛选结果
    if hasattr(step_corr, 'to_drop') and step_corr.to_drop is not None:
        drop_indices = set(step_corr.to_drop)
        feature_names = [name for i, name in enumerate(feature_names) if i not in drop_indices]
        n_after_corr = len(feature_names)
        if verbose:
            print(f"相关性过滤后剩余特征数: {n_after_corr}")
    else:
        n_after_corr = n_after_variance
        if verbose:
            print("警告: CorrelationFilter 未记录删除的列，可能未被拟合。")

    # 3. 检查模型是否支持特征重要性
    if not hasattr(step_model, 'feature_importances_'):
        if verbose:
            print(f"当前模型 {type(step_model).__name__} 不支持 feature_importances_ 属性。")
        return None

    importances = step_model.feature_importances_

    # 4. 检查维度匹配情况
    if len(importances) != len(feature_names):
        raise ValueError(f"维度不匹配：特征名数量({len(feature_names)}) 与 重要性得分数量({len(importances)}) 不一致！")

    # 5. 构建并排序 DataFrame
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # 6. 可视化
    if show_plot:
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            hue='Feature',
            data=feature_imp_df.head(plot_top_n), 
            palette='viridis',
            legend=False
        )
        plt.title(f'Top {plot_top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    # 7. 打印统计信息
    if verbose:
        print(f"\n=== 特征重要性 Top {min(20, len(feature_imp_df))} ===")
        print(feature_imp_df.head(20))
        print(f"\n特征工程已完成，已提取前 {top_n} 个重要特征。")

    # 8. 返回结果字典
    return {
        'feature_importance_df': feature_imp_df,
        'top_features': feature_imp_df['Feature'].head(top_n).tolist(),
        'final_feature_names': feature_names,
        'n_features_after_variance': n_after_variance,
        'n_features_after_correlation': n_after_corr
    }

# ================= 使用示例 =================
if __name__ == "__main__":
    try:
        # 尝试加载数据（用户提供的路径）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '../data/Mordred_descriptors_data_PCE_revised.csv')
        df = pd.read_csv(data_path)
        print(df.head())
    except Exception:
        # 如果失败，生成一些虚拟数据用于测试
        print("未找到指定路径数据，正在生成虚拟数据用于演示...")
        data = np.random.exponential(scale=2.0, size=(100, 5))
        df = pd.DataFrame(data, columns=[f'feat_{i}' for i in range(5)])
        df['PCE'] = np.random.rand(100)

    # 数据预清洗
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    print(df.head())
    print("Current df columns:", df.columns.tolist())        
    
    y = df['PCE']
    X = df.drop(columns=['PCE']).apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.isnull().mean() <= 0.5]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_rf = train_and_evaluate_multi_model(X_train, X_test, y_train, y_test, model_type='rf')
    target_pipeline = results_rf['best_model']

    results_rf_analysis = analyze_feature_importance(
        pipeline=target_pipeline,
        original_feature_names=X.columns,
        top_n=50
    )

    rf_top_50_features = results_rf_analysis['top_features']