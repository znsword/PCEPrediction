import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predicted_vs_true(y_true, y_pred, title='MLP Model: Predicted vs. True Values', save_path=None):
    """
    绘制预测值与真实值的散点图。
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    
    # 绘制对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    plt.show()

def plot_model_comparison(performance_dict, metric='R2', title='Model Performance Comparison', save_path=None):
    """
    对比不同模型的性能指标（如 R2 分数）。
    """
    model_names = list(performance_dict.keys())
    metric_values = [metrics[metric] for metrics in performance_dict.values()]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=metric_values, palette='viridis', hue=model_names, legend=False)
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric)
    
    # 设置 y 轴范围，确保显示美观
    min_y = min(0, min(metric_values) - 0.1)
    max_y = max(1, max(metric_values) + 0.1)
    plt.ylim(min_y, max_y)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在条形上方显示数值
    for index, value in enumerate(metric_values):
        plt.text(index, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
        
    if save_path:
        plt.savefig(save_path)
        print(f"对比图已保存至: {save_path}")
    plt.show()

def plot_pca_variance(cum_variance, target_variance, best_n, save_path=None):
    """
    绘制 PCA 累计解释方差图。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--')
    plt.axhline(y=target_variance, color='r', linestyle='-', label=f'Target {target_variance*100}%')
    plt.axvline(x=best_n, color='g', linestyle='-', label=f'Best n={best_n}')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"PCA 碎石图已保存至: {save_path}")
    plt.show()
