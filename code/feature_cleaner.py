import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from enum import Enum
from typing import List, Optional, Union, Literal, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import (
    PowerTransformer, 
    StandardScaler, 
    QuantileTransformer, 
    FunctionTransformer
)
from sklearn.pipeline import Pipeline

# 配置日志和绘图样式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试设置中文字体，如果环境不支持则回退到默认
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
sns.set_style("whitegrid")

class TransformMethod(Enum):
    """数据变换方法枚举"""
    NONE = "none"                    # 不进行变换
    YEO_JOHNSON = "yeo-johnson"      # Yeo-Johnson 变换（默认，支持正负值）
    BOX_COX = "box-cox"              # Box-Cox 变换（要求数据严格大于 0）
    LOG = "log"                      # 自然对数 log(x)
    LOG1P = "log1p"                  # log(1+x) 变换（适用于含有较多 0 的数据）
    QUANTILE_NORMAL = "quantile-normal"    # 分位数变换（映射至正态分布，对异常值鲁棒）
    QUANTILE_UNIFORM = "quantile-uniform"  # 分位数变换（映射至均匀分布）

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    自定义转换器：移除彼此高度相关的特征。
    """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.to_drop_: List[str] = []
        self.feature_names_in_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.values
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.drop(columns=self.to_drop_)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            raise RuntimeError("Transformer 尚未 fit")
        return [f for f in self.feature_names_in_ if f not in self.to_drop_]


class FeatureCleaner:
    """
    特征清洗器：封装完整的数据预处理流程，支持可选的数据变换方式。
    """
    
    def __init__(
        self,
        impute_strategy: str = 'median',
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.9,
        transform_method: Union[str, TransformMethod] = TransformMethod.YEO_JOHNSON,
        enable_scaling: bool = True,
        verbose: bool = True
    ):
        self.impute_strategy = impute_strategy
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.transform_method = self._parse_transform_method(transform_method)
        self.enable_scaling = enable_scaling
        self.verbose = verbose
        
        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False

    def _parse_transform_method(self, method: Union[str, TransformMethod]) -> TransformMethod:
        if isinstance(method, TransformMethod):
            return method
        try:
            return TransformMethod(method.lower())
        except (ValueError, AttributeError):
            valid_options = [m.value for m in TransformMethod]
            raise ValueError(f"不支持的变换方法: {method}。可选值: {valid_options}")

    def _get_transformer_step(self) -> Optional[BaseEstimator]:
        m = self.transform_method
        if m == TransformMethod.NONE:
            return None
        elif m == TransformMethod.YEO_JOHNSON:
            return PowerTransformer(method='yeo-johnson')
        elif m == TransformMethod.BOX_COX:
            return PowerTransformer(method='box-cox')
        elif m == TransformMethod.LOG:
            return FunctionTransformer(np.log, inverse_func=np.exp, validate=True, check_inverse=False)
        elif m == TransformMethod.LOG1P:
            return FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True, check_inverse=False)
        elif m == TransformMethod.QUANTILE_NORMAL:
            return QuantileTransformer(output_distribution='normal', random_state=42)
        elif m == TransformMethod.QUANTILE_UNIFORM:
            return QuantileTransformer(output_distribution='uniform', random_state=42)
        return None

    def _log(self, message: str):
        if self.verbose:
            logger.info(message)

    def _build_pipeline(self) -> Pipeline:
        set_config(transform_output="pandas")
        steps = [
            ('imputer', SimpleImputer(strategy=self.impute_strategy)),
            ('var_filter', VarianceThreshold(threshold=self.variance_threshold)),
            ('corr_filter', CorrelationFilter(threshold=self.correlation_threshold)),
        ]
        transformer = self._get_transformer_step()
        if transformer is not None:
            steps.append(('data_transform', transformer))
        if self.enable_scaling:
            steps.append(('scaler', StandardScaler()))
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'FeatureCleaner':
        self._log(f"开始拟合清洗管道... 原始特征数: {X.shape[1]}")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureCleaner 尚未拟合")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        if not self._is_fitted:
            return []
        return self.pipeline.get_feature_names_out().tolist()

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'FeatureCleaner':
        return joblib.load(filepath)


class TransformVisualizer:
    """
    数据变换可视化器：提供多种图表对比变换前后的数据分布。
    """
    def __init__(self, figsize_base: tuple = (5, 4), palette: str = 'husl'):
        self.colors = sns.color_palette(palette, 10)
    
    def plot_single_feature_comparison(self, original: pd.Series, transformed: pd.Series, feature_name: str, method_name: str = "Transformed") -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # 1. 原始直方图
        sns.histplot(original, kde=True, ax=axes[0, 0], color=self.colors[0], alpha=0.7)
        axes[0, 0].set_title(f'Original: {feature_name}', fontsize=12, fontweight='bold')
        skew_orig = stats.skew(original.dropna())
        axes[0, 0].text(0.95, 0.95, f'Skew: {skew_orig:.3f}', transform=axes[0, 0].transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. 变换后直方图
        sns.histplot(transformed, kde=True, ax=axes[0, 1], color=self.colors[1], alpha=0.7)
        axes[0, 1].set_title(f'{method_name} Transformed: {feature_name}', fontsize=12, fontweight='bold')
        skew_trans = stats.skew(transformed.dropna())
        axes[0, 1].text(0.95, 0.95, f'Skew: {skew_trans:.3f}', transform=axes[0, 1].transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # 3. 原始 QQ
        stats.probplot(original.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Original Q-Q Plot')
        
        # 4. 变换后 QQ
        stats.probplot(transformed.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{method_name} Q-Q Plot')
        
        plt.tight_layout()
        return fig
    
    def plot_multi_method_comparison(self, original_df: pd.DataFrame, transformed_dict: Dict[str, pd.DataFrame], feature_name: str) -> plt.Figure:
        n_methods = len(transformed_dict)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=((n_methods + 1) * 4, 8))
        
        orig_data = original_df[feature_name].dropna()
        sns.histplot(orig_data, kde=True, ax=axes[0, 0], color='gray', alpha=0.7)
        stats.probplot(orig_data, dist="norm", plot=axes[1, 0])
        axes[0, 0].set_title('Original', fontweight='bold')
        
        for i, (method_name, trans_df) in enumerate(transformed_dict.items()):
            if feature_name in trans_df.columns:
                data = trans_df[feature_name].dropna()
                sns.histplot(data, kde=True, ax=axes[0, i+1], color=self.colors[i], alpha=0.7)
                stats.probplot(data, dist="norm", plot=axes[1, i+1])
                axes[0, i+1].set_title(f'{method_name}\nSkew: {stats.skew(data):.2f}', fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_skewness_comparison(self, original_df: pd.DataFrame, transformed_dict: Dict[str, pd.DataFrame], top_n: int = 15) -> plt.Figure:
        orig_skew = original_df.apply(lambda x: stats.skew(x.dropna())).sort_values(key=abs, ascending=False)
        top_features = orig_skew.head(top_n).index.tolist()
        
        data_list = []
        for feat in top_features:
            data_list.append({'Feature': feat, 'Method': 'Original', 'AbsSkewness': abs(orig_skew[feat])})
            for m_name, t_df in transformed_dict.items():
                if feat in t_df.columns:
                    s = stats.skew(t_df[feat].dropna())
                    data_list.append({'Feature': feat, 'Method': m_name, 'AbsSkewness': abs(s)})
        
        df_plot = pd.DataFrame(data_list)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_plot, x='Feature', y='AbsSkewness', hue='Method', ax=ax, palette='Set2')
        ax.axhline(y=0.5, color='red', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        ax.set_title('Skewness Comparison (Abs Error)', fontweight='bold')
        plt.tight_layout()
        return fig

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
    
    # 构建对比
    cleaner_yj = FeatureCleaner(transform_method='yeo-johnson').fit(X_train)
    X_yj = cleaner_yj.transform(X_train)
    
    cleaner_qn = FeatureCleaner(transform_method='quantile-normal').fit(X_train)
    X_qn = cleaner_qn.transform(X_train)
    
    cleaner_none = FeatureCleaner(transform_method='none').fit(X_train)
    X_none = cleaner_none.transform(X_train)

    # 启动可视化
    viz = TransformVisualizer()
    common_feat = list(set(X_train.columns) & set(X_yj.columns))
    if common_feat:
        sample = common_feat[0]
        # 1. 单特征对比图
        viz.plot_single_feature_comparison(X_train[sample], X_yj[sample], sample, "Yeo-Johnson")
        
        # 2. 多方法对比图
        transformed_dict = {'Yeo-Johnson': X_yj, 'Quantile-Normal': X_qn, 'None': X_none}
        viz.plot_multi_method_comparison(X_train, transformed_dict, sample)
        
        # 3. 整体偏度对比图
        viz.plot_skewness_comparison(X_train, transformed_dict)
        
        plt.show()
    else:
        print("无共同特征可绘制。")