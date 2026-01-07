import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from itertools import product
from .models import MLPRegressor

def train_mlp(model, dataloader, criterion, optimizer, num_epochs, verbose=False):
    """
    执行 MLP 模型的训练循环。
    """
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        current_epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            current_epoch_loss += loss.item()
        
        avg_loss = current_epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
    return train_losses

def evaluate_mlp(model, X_test_tensor, y_test_tensor):
    """
    在测试集上评估 MLP 模型。
    返回 MAE, RMSE, R2 分数。
    """
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    y_test_np = y_test_tensor.cpu().numpy().flatten()
    y_pred_np = y_pred_tensor.cpu().numpy().flatten()
    
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    r2 = r2_score(y_test_np, y_pred_np)
    
    return mae, rmse, r2

def train_and_evaluate_mlp_cv(
    X_tensor, y_tensor, input_dim, output_dim, 
    hidden_dim, learning_rate, num_epochs, batch_size, n_splits=5
):
    """
    使用 K 折交叉验证训练和评估 MLP 模型。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes, fold_rmses, fold_r2s = [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X_tensor)):
        X_train_fold, X_val_fold = X_tensor[train_index], X_tensor[val_index]
        y_train_fold, y_val_fold = y_tensor[train_index], y_tensor[val_index]

        dataset = TensorDataset(X_train_fold, y_train_fold)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = MLPRegressor(input_dim, hidden_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_mlp(model, dataloader, criterion, optimizer, num_epochs)
        
        mae, rmse, r2 = evaluate_mlp(model, X_val_fold, y_val_fold)
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

    return np.mean(fold_maes), np.mean(fold_rmses), np.mean(fold_r2s)

def tune_mlp_hyperparameters(
    X_tensor, y_tensor, input_dim, output_dim,
    hidden_dims, learning_rates, num_epochs_list, batch_sizes, n_splits=5
):
    """
    对 MLP 模型执行网格搜索式的超参数调优。
    """
    best_r2 = -float('inf')
    best_params = {}
    results = []

    param_combinations = list(product(hidden_dims, learning_rates, num_epochs_list, batch_sizes))
    print(f"开始 MLP 超参数调优，共 {len(param_combinations)} 种组合...")

    for i, (h_dim, lr, epochs, b_size) in enumerate(param_combinations):
        avg_mae, avg_rmse, avg_r2 = train_and_evaluate_mlp_cv(
            X_tensor, y_tensor, input_dim, output_dim,
            h_dim, lr, epochs, b_size, n_splits
        )
        
        results.append({
            'params': {'hidden_dim': h_dim, 'learning_rate': lr, 'num_epochs': epochs, 'batch_size': b_size},
            'avg_r2': avg_r2
        })

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_params = {'hidden_dim': h_dim, 'learning_rate': lr, 'num_epochs': epochs, 'batch_size': b_size}
            print(f"组合 {i+1}/{len(param_combinations)}: 新的最佳 R2 = {best_r2:.4f}")

    return best_params, best_r2, results
