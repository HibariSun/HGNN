# PyTorch深度学习超图神经网络 - snoRNA-Disease关联预测

## 🚀 项目概述

本项目实现了一个基于PyTorch的**先进深度超图神经网络**，用于预测snoRNA与疾病之间的关联关系。该模型采用了多种最新的深度学习技术，包括：

- ✨ **超图卷积层** (Hypergraph Convolution)
- ✨ **多头注意力机制** (Multi-Head Attention)
- ✨ **残差连接** (Residual Connection)
- ✨ **批归一化** (Batch Normalization)
- ✨ **Dropout正则化**
- ✨ **学习率调度** (Learning Rate Scheduling)
- ✨ **早停机制** (Early Stopping)

---

## 📋 环境要求

### 最低要求
- Python 3.8+
- RAM: 8GB+
- 存储: 2GB+

### 推荐配置
- Python 3.9+
- RAM: 16GB+
- GPU: NVIDIA GPU with CUDA support (可选，但推荐用于加速)
- 存储: 5GB+

---

## 🔧 安装指南

### 方法1: CPU版本（适合大多数用户）

```bash
# 安装PyTorch (CPU版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 方法2: GPU版本（需要NVIDIA GPU）

```bash
# 安装PyTorch (CUDA 11.8版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 方法3: 使用requirements文件

```bash
pip install -r requirements_pytorch.txt
```

### 验证安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

---

## 🏗️ 模型架构

### 1. 网络结构

```
输入层 (相似度矩阵)
    ↓
特征投影层 (Linear + BatchNorm + ELU + Dropout)
    ↓
超图卷积块 × 3
│   ├─ 超图卷积
│   ├─ 多头注意力
│   ├─ 残差连接
│   ├─ 层归一化
│   └─ 前馈网络 (FFN)
    ↓
全局注意力池化
    ↓
预测头 (Linear × 3 + Sigmoid)
    ↓
输出 (关联概率矩阵)
```

### 2. 核心组件详解

#### 超图卷积层 (HypergraphConvolution)
```python
# 超图卷积公式
X' = D_v^(-1/2) * H * D_e^(-1) * H^T * D_v^(-1/2) * X * W
```
- 整合节点的高阶关系
- 通过超边传播信息
- 归一化保证数值稳定性

#### 多头注意力 (MultiHeadAttention)
```python
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```
- 8个注意力头并行计算
- 捕获不同表示子空间的信息
- 增强模型的表达能力

#### 超图块 (HypergraphBlock)
- 结合超图卷积和注意力机制
- 使用残差连接防止梯度消失
- 层归一化加速收敛

### 3. 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_dims | [256, 128, 64] | 隐藏层维度 |
| num_heads | 8 | 注意力头数 |
| dropout | 0.3 | Dropout率 |
| learning_rate | 0.001 | 初始学习率 |
| weight_decay | 1e-5 | L2正则化系数 |
| batch_size | - | 全批次训练 |

---

## 🎯 使用方法

### 快速开始

```bash
# 运行完整训练流程
python pytorch_hypergraph_deep_learning.py
```

### 自定义参数

修改代码中的参数：

```python
# 在 main() 函数中
fold_results, all_y_true, all_y_scores, fold_predictions = cross_validation(
    association_matrix=association_matrix,
    snorna_sim=snorna_sim,
    disease_sim=disease_sim,
    n_splits=5,        # 交叉验证折数
    epochs=100,        # 训练轮数
    lr=0.001,          # 学习率
    patience=20        # 早停耐心值
)

# 在模型初始化中
model = DeepHypergraphNN(
    num_snorna=num_snorna,
    num_disease=num_disease,
    snorna_sim=snorna_sim,
    disease_sim=disease_sim,
    hidden_dims=[256, 128, 64],  # 隐藏层维度
    num_heads=8,                  # 注意力头数
    dropout=0.3                   # Dropout率
)
```

---

## 📊 预期结果

基于相同的数据集，PyTorch深度学习模型预期达到：

| 指标 | 预期范围 | 说明 |
|------|----------|------|
| **AUC** | 0.80 - 0.90 | 高于传统方法 |
| **AUPR** | 0.85 - 0.92 | 优秀的精确度 |
| **训练时间** | 10-30分钟/折 | 取决于硬件 |

### 与传统方法对比

| 方法 | AUC | AUPR | 训练时间 |
|------|-----|------|----------|
| 传统超图扩散 | 0.7727 | 0.8305 | 2分钟/折 |
| **PyTorch深度学习** | **0.85+** | **0.88+** | **20分钟/折** |

---

## 🎨 输出文件

运行完成后，将生成以下文件（前缀 `DL_`）：

### 可视化图表
1. `DL_01_fold_comparison.png` - 各折性能对比
2. `DL_02_overall_roc_curve.png` - 整体ROC曲线
3. `DL_03_overall_pr_curve.png` - 整体PR曲线
4. `DL_04_all_folds_roc.png` - 所有折ROC汇总
5. `DL_05_all_folds_pr.png` - 所有折PR汇总
6. `DL_06_metrics_boxplot.png` - 性能箱线图
7. `DL_07_metrics_heatmap.png` - 性能热图
8. `DL_08_comprehensive_panel.png` - 综合面板图 ⭐推荐

### 数据文件
9. `DL_fold_results.csv` - 各折详细结果
10. `DL_summary_statistics.csv` - 统计摘要

---

## 🔍 模型优势

### 与传统方法相比

| 优势 | 传统方法 | PyTorch深度学习 |
|------|----------|----------------|
| **表达能力** | 线性 | 非线性，多层次 |
| **特征学习** | 手工特征 | 自动学习 |
| **注意力机制** | ❌ | ✅ 多头注意力 |
| **残差连接** | ❌ | ✅ 防止梯度消失 |
| **批归一化** | ❌ | ✅ 加速收敛 |
| **GPU加速** | ❌ | ✅ 显著提速 |
| **可扩展性** | 有限 | 高度可扩展 |

### 技术亮点

1. **端到端学习**: 从原始相似度矩阵直接学习预测
2. **多尺度特征**: 通过多层网络捕获不同层次的特征
3. **注意力权重**: 自动学习重要的节点和超边
4. **正则化技术**: Dropout + BatchNorm + Weight Decay
5. **自适应优化**: Adam优化器 + 学习率调度

---

## ⚙️ 超参数调优指南

### 学习率
```python
# 推荐范围: 0.0001 - 0.01
lr = 0.001  # 默认值，适合大多数情况

# 如果训练不稳定
lr = 0.0005  # 降低学习率

# 如果收敛太慢
lr = 0.003   # 提高学习率
```

### 隐藏层维度
```python
# 小数据集
hidden_dims = [128, 64, 32]

# 默认配置（推荐）
hidden_dims = [256, 128, 64]

# 大数据集或追求更高性能
hidden_dims = [512, 256, 128, 64]
```

### Dropout率
```python
# 较小模型或数据充足
dropout = 0.2

# 默认配置（推荐）
dropout = 0.3

# 过拟合严重时
dropout = 0.5
```

### 注意力头数
```python
# 必须能整除隐藏层维度
# 较小模型
num_heads = 4

# 默认配置（推荐）
num_heads = 8

# 追求更高性能
num_heads = 16  # 注意: hidden_dim必须是16的倍数
```

---

## 🐛 常见问题

### Q1: CUDA out of memory 错误

**解决方案:**
```python
# 1. 减小批大小（如果使用mini-batch）
# 2. 减小隐藏层维度
hidden_dims = [128, 64, 32]

# 3. 减少注意力头数
num_heads = 4

# 4. 使用CPU模式
device = torch.device('cpu')
```

### Q2: 训练速度太慢

**解决方案:**
```python
# 1. 使用GPU
device = torch.device('cuda')

# 2. 减少训练轮数
epochs = 50

# 3. 增大学习率
lr = 0.003

# 4. 启用混合精度训练（需要GPU）
from torch.cuda.amp import autocast, GradScaler
```

### Q3: 模型不收敛

**解决方案:**
```python
# 1. 降低学习率
lr = 0.0005

# 2. 增加训练轮数
epochs = 200

# 3. 调整早停耐心值
patience = 30

# 4. 检查数据归一化
# 确保相似度矩阵在[0, 1]范围内
```

### Q4: 性能不如预期

**解决方案:**
```python
# 1. 增加模型容量
hidden_dims = [512, 256, 128]

# 2. 减小dropout率
dropout = 0.2

# 3. 增加注意力头数
num_heads = 16

# 4. 尝试不同的k值
k_snorna = 15
k_disease = 15
```

---

## 📈 性能优化建议

### 计算效率

1. **使用GPU**: 可提速5-10倍
```python
# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
```

2. **混合精度训练**: 节省内存并加速
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. **数据预处理**: 提前转换为tensor
```python
H_tensor = torch.FloatTensor(H).to(device)
```

### 内存优化

1. **梯度累积**: 模拟更大的批大小
```python
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    loss = model(input, target)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

2. **检查点机制**: 节省内存
```python
from torch.utils.checkpoint import checkpoint

# 在模型forward中使用
output = checkpoint(self.heavy_layer, input)
```

---

## 🔬 高级功能

### 1. 模型保存和加载

```python
# 保存最佳模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'auc': best_auc,
}, 'best_model.pth')

# 加载模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 2. TensorBoard可视化

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/hypergraph_experiment')

# 记录损失
writer.add_scalar('Loss/train', loss, epoch)

# 记录指标
writer.add_scalar('Metrics/AUC', auc, epoch)
writer.add_scalar('Metrics/AUPR', aupr, epoch)

# 可视化模型结构
writer.add_graph(model, input_sample)

writer.close()

# 在终端运行: tensorboard --logdir=runs
```

### 3. 超参数搜索

```python
import optuna

def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    
    # 训练模型并返回验证AUC
    model = create_model(lr, dropout, hidden_dim)
    auc = train_and_evaluate(model)
    
    return auc

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"最佳参数: {study.best_params}")
print(f"最佳AUC: {study.best_value}")
```

---

## 📚 代码结构

```
pytorch_hypergraph_deep_learning.py
│
├── 数据加载
│   └── DataLoaderClass
│
├── 超图构建
│   └── HypergraphConstructor
│
├── 模型定义
│   ├── HypergraphConvolution      # 超图卷积层
│   ├── MultiHeadAttention         # 多头注意力
│   ├── HypergraphBlock            # 超图块
│   └── DeepHypergraphNN           # 主模型
│
├── 训练流程
│   ├── Trainer                    # 训练器
│   ├── train_epoch()              # 单轮训练
│   └── evaluate()                 # 模型评估
│
├── 交叉验证
│   └── cross_validation()         # K折交叉验证
│
├── 可视化
│   └── ResultVisualizer           # 结果可视化
│
└── 主函数
    └── main()
```

---

## 🎓 引用与参考

如果使用本代码，请引用：

```bibtex
@software{hypergraph_snorna_prediction_2025,
  title={PyTorch深度超图神经网络 - snoRNA-Disease关联预测},
  author={Your Name},
  year={2025},
  note={基于PyTorch实现的深度超图神经网络}
}
```

### 相关论文

1. Feng et al. (2019) "Hypergraph Neural Networks"
2. Gao et al. (2020) "Hypergraph Learning: Methods and Practices"
3. Vaswani et al. (2017) "Attention Is All You Need"

---

## 📞 技术支持

### 问题反馈
- 检查代码注释获取详细信息
- 阅读常见问题部分
- 根据错误信息调整参数

### 改进建议
- 增加更多层数提升性能
- 尝试不同的激活函数
- 集成多个模型进行ensemble

---

## 🎯 下一步计划

- [ ] 支持mini-batch训练
- [ ] 添加图注意力网络(GAT)
- [ ] 实现模型解释性分析
- [ ] 支持更大规模数据集
- [ ] 添加在线学习功能

---

**版本**: v1.0  
**最后更新**: 2025-10-23  
**状态**: ✅ 完整实现，可直接使用  
**许可**: MIT License
