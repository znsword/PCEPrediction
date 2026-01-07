import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    """
    多层感知机 (MLP) 回归模型。
    
    参数:
        input_dim (int): 输入层的维度（特征数量）。
        hidden_dim (int): 隐藏层的维度。
        output_dim (int): 输出层的维度（回归任务通常为1）。
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播过程。
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
