# 代码功能模块分析

|模块|功能描述|
|---|---|
|数据加载|从CSV文件加载数据，分离特征和目标变量|
|数据清洗|处理缺失值、方差过滤、相关性过滤、标准化|
|PCA降维|自动主成分分析降维|
|MLP模型|PyTorch MLP回归器定义|
|训练与评估|模型训练、交叉验证、超参数调优|
|可视化|结果可视化|

# 文件结构

```bash
PCEPrediction/
├── code/
│   ├── __init__.py
│   ├── config.py          # 配置参数
│   ├── data_utils.py      # 数据加载与清洗
│   ├── models.py          # 模型定义
│   ├── training.py        # 训练与评估函数
│   └── visualization.py   # 可视化函数
└── run_mlp_pipeline.py    # 主执行脚本
```

# 代码文件说明

|文件名|内容描述|
|---|---|
|config.py|配置文件，包含数据路径、目标变量、清洗参数、PCA参数、模型参数等|
|data_utils.py|数据处理模块，包含数据加载、清洗、PCA降维等函数|
|models.py|模型定义模块，包含MLP回归器 `MLPRegressor` 类定义|
|training.py|训练模块，包含模型训练、交叉验证、超参数调优等函数|
|visualization.py|可视化模块，包含结果可视化函数|
|run_mlp_pipeline.py|主流程：加载数据->清洗->PCA->训练->评估|
