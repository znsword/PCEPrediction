# 数据清洗

为了确保在处理 **128个样本、1120个特征** 的高维数据时，特征（X）与标签（y）始终保持完美对应，数据清洗必须遵循“**先合并、后清洗、再分离**”的核心原则。

以下是为您总结的详细清洗流程：

---

## 🛠️ 数据清洗完整流程图

### 第一步：同步准备 (Synchronization)

在进行任何过滤之前，必须确保特征矩阵和标签向量的顺序是一致的。

1. **加载数据**：将 Mordred 描述符和活性标签分别载入 `df_X` 和 `df_y`。
2. **强制对齐**：使用 `pd.concat` 将两者按列合并。

```python
# 将特征和标签合并为一个 DataFrame，确保行索引 (Index) 锁定对应关系
df_combined = pd.concat([df_X, df_y], axis=1)

```

---

### 第二步：行级别清洗 (Row-wise Cleaning)

这一步会删除不合格的样本。由于特征和标签已合并，删除行时两者会同步消失。

1. **标签缺失值处理**：如果活性值（Label）缺失，该样本必须删除。
2. **错误码处理**：将 Mordred 中的非数值错误字符串（如 "Can't compute"）转换为 `NaN`。
3. **样本去重**：删除完全重复的分子行。

```python
# 转换非数值字符为 NaN
df_combined = df_combined.apply(pd.to_numeric, errors='coerce')

# 删除标签列为空的行
df_combined = df_combined.dropna(subset=[label_column_name])

```

---

### 第三步：特征级别清洗 (Feature-wise Cleaning)

这一步处理 1120 个描述符，但**不改变样本行数**，因此不会破坏对应关系。

1. **缺失率过滤**：删除缺失值占比超过 30% 的特征列。
2. **方差过滤**：使用 `VarianceThreshold` 删除常量列（所有分子取值都一样的特征）。
3. **中位数填充**：对剩余特征中少量的 `NaN` 进行填充（推荐中位数）。

```python
from sklearn.feature_selection import VarianceThreshold

# 仅对特征部分进行方差过滤
selector = VarianceThreshold(threshold=0)
X_temp = df_combined.drop(columns=[label_column_name])
selector.fit(X_temp)

# 保留下来的特征名
selected_cols = X_temp.columns[selector.get_support()]
df_combined = df_combined[list(selected_cols) + [label_column_name]]

```

---

### 第四步：数值变换与分离 (Transformation & Splitting)

在进行 PCA 降维和标准化之前，必须将特征和标签再次分离。

1. **标准化**：对特征进行 `StandardScaler` 处理。
2. **降维 (PCA)**：提取主成分。
3. **重建 DataFrame**：**最关键的一步**，将 PCA 的结果转回 DataFrame 时，必须手动赋值原始的 Index。

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 再次分离
X_final = df_combined.drop(columns=[label_column_name])
y_final = df_combined[label_column_name]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# PCA 降维
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 【核心步骤】：通过 index=y_final.index 确保标签对应关系
df_pca = pd.DataFrame(X_pca, index=y_final.index, 
                      columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

```

---

## 📈 清洗前后对应关系核对表

| 环节 | 特征 (X) 状态 | 标签 (y) 状态 | 如何保证对应？ |
| --- | --- | --- | --- |
| **合并期** | 1120 维 | 1 维 | 封装在同一个 `df_combined` 中。 |
| **删行期** | 样本数减少 | 样本数同步减少 | `dropna` 或 `drop` 作用于整个 DataFrame。 |
| **填充期** | 特征列变整洁 | 保持不变 | 不涉及行移动。 |
| **PCA 降维** | 变为  个 PC | 保持不变 | **手动将 y 的 Index 赋给 PCA 后的 DataFrame。** |

---

## 🏁 总结

要实现完美对应，只需记住一句话：**在“删行”阶段，把 X 和 y 绑在一起操作；在“降维”阶段，用原始的 Index 标签把 PCA 的结果重新包装起来。**

通过这种方法，即使你最终得到的是  的主成分矩阵，它的每一行依然能精准对应到原始分子的活性值。

**下一步：**
既然清洗和降维已经完成且对应完美，您是否准备好进入**模型训练**阶段？我们可以先从**特征重要性分析**开始，看看降维后的哪些主成分对您的活性标签影响最大。需要我为您写出这段分析代码吗？