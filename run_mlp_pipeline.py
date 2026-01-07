import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from code.config import (
    DATA_PATH, TARGET_COLUMN, CLEANING_PARAMS, 
    PCA_TARGET_VARIANCE, DEFAULT_MODEL_PARAMS, 
    RANDOM_STATE, TEST_SIZE
)
from code.data_utils import load_data, clean_data_manual, auto_pca_reduction
from code.models import MLPRegressor
from code.training import train_mlp, evaluate_mlp
from code.visualization import plot_predicted_vs_true

def run_pipeline():
    """
    运行完整的 MLP 预测工作流。
    """
    # 1. 数据加载
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 数据文件未找到: {DATA_PATH}")
        return

    X, y = load_data(DATA_PATH, target_column=TARGET_COLUMN)

    # 2. 数据清洗 (包含标准化)
    X_clean, y_clean = clean_data_manual(X, y, **CLEANING_PARAMS)

    # 3. PCA 降维
    X_reduced, pca_model = auto_pca_reduction(X_clean, target_variance=PCA_TARGET_VARIANCE)

    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5. 转换为 PyTorch 张量
    # 注意：使用 .values 确保去掉索引，只取数值
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # 6. 准备数据加载器
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=DEFAULT_MODEL_PARAMS['batch_size'], shuffle=True)

    # 7. 初始化模型、优化器和损失函数
    input_dim = X_reduced.shape[1]
    model = MLPRegressor(
        input_dim=input_dim, 
        hidden_dim=DEFAULT_MODEL_PARAMS['hidden_dim'], 
        output_dim=DEFAULT_MODEL_PARAMS['output_dim']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_MODEL_PARAMS['learning_rate'])
    criterion = nn.MSELoss()

    # 8. 模型训练
    print("\n🚀 开始训练 MLP 模型...")
    train_mlp(
        model, 
        dataloader, 
        criterion, 
        optimizer, 
        num_epochs=DEFAULT_MODEL_PARAMS['num_epochs'], 
        verbose=True
    )

    # 9. 模型评估
    print("\n📊 评估模型在测试集上的表现...")
    mae, rmse, r2 = evaluate_mlp(model, X_test_tensor, y_test_tensor)
    print(f"🔹 测试集指标:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")

    # 10. 结果可视化
    print("\n🎨 生成可视化图表...")
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    plot_predicted_vs_true(
        y_test_tensor.cpu().numpy().flatten(), 
        y_pred_tensor.cpu().numpy().flatten(),
        title='MLP Results: Predicted vs. True PCE'
    )

if __name__ == "__main__":
    run_pipeline()
