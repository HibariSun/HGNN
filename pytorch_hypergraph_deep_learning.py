"""
基于PyTorch的深度超图神经网络 - snoRNA-Disease关联预测
包含注意力机制、残差连接、批归一化等先进技术
使用5折交叉验证和完整的性能评估
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            roc_curve, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*80)
print(" 基于PyTorch的深度超图神经网络 - snoRNA-Disease关联预测")
print("="*80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


class HypergraphConvolution(nn.Module):
    """超图卷积层"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(HypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, X, H):
        """
        X: 节点特征矩阵 (num_nodes, in_features)
        H: 超图关联矩阵 (num_nodes, num_hyperedges)
        """
        # 计算度矩阵
        D_v = torch.sum(H, dim=1, keepdim=True).clamp(min=1)  # 节点度
        D_e = torch.sum(H, dim=0, keepdim=True).clamp(min=1)  # 超边度
        
        # 归一化
        H_norm = H / torch.sqrt(D_v)
        H_norm = H_norm / torch.sqrt(D_e)
        
        # 超图卷积: X' = D_v^(-1/2) H D_e^(-1) H^T D_v^(-1/2) X W
        X_transformed = self.linear(X)
        X_aggregated = H_norm @ (H_norm.T @ X_transformed)
        
        # 批归一化
        X_aggregated = self.bn(X_aggregated)
        
        return X_aggregated


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 线性变换并分割成多头
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # 输出变换
        x = self.out_linear(x)
        
        return x


class HypergraphBlock(nn.Module):
    """超图块（包含超图卷积、注意力、残差连接）"""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3):
        super(HypergraphBlock, self).__init__()
        
        self.hgc = HypergraphConvolution(in_features, out_features)
        self.attention = MultiHeadAttention(out_features, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features),
            nn.Dropout(dropout)
        )
        
        # 残差连接的投影层
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, X, H):
        # 保存输入用于残差连接
        identity = self.residual_proj(X)
        
        # 超图卷积
        X = self.hgc(X, H)
        X = F.elu(X)
        X = self.norm1(X + identity)
        
        # 注意力机制
        X_att = self.attention(X.unsqueeze(0)).squeeze(0)
        X = self.norm2(X + X_att)
        
        # 前馈网络
        X = X + self.ffn(X)
        
        return X


class DeepHypergraphNN(nn.Module):
    """深度超图神经网络"""
    
    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim,
                 hidden_dims=[256, 128, 64], num_heads=8, dropout=0.3):  # 原始参数值hidden_dims=[256, 128, 64], num_heads=8, dropout=0.3
        super(DeepHypergraphNN, self).__init__()
        
        self.num_snorna = num_snorna
        self.num_disease = num_disease
        
        # 将相似度矩阵转换为参数（可学习）
        self.snorna_features = nn.Parameter(torch.FloatTensor(snorna_sim), requires_grad=True)
        self.disease_features = nn.Parameter(torch.FloatTensor(disease_sim), requires_grad=True)
        
        # 特征投影
        self.snorna_projection = nn.Sequential(
            nn.Linear(num_snorna, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        self.disease_projection = nn.Sequential(
            nn.Linear(num_disease, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # 超图卷积块
        self.hg_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hg_blocks.append(
                HypergraphBlock(dims[i], dims[i+1], num_heads, dropout)
            )
        
        # 全局注意力池化
        self.global_attention = MultiHeadAttention(hidden_dims[-1], num_heads, dropout)
        
        # 预测头
        final_dim = hidden_dims[-1]
        self.predictor = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, H):
        """前向传播"""
        # 投影初始特征
        snorna_feat = self.snorna_projection(self.snorna_features)
        disease_feat = self.disease_projection(self.disease_features)
        
        # 拼接所有节点特征
        X = torch.cat([snorna_feat, disease_feat], dim=0)
        
        # 通过超图卷积块
        for hg_block in self.hg_blocks:
            X = hg_block(X, H)
        
        # 全局注意力
        X = self.global_attention(X.unsqueeze(0)).squeeze(0)
        
        # 分离特征
        snorna_embed = X[:self.num_snorna]
        disease_embed = X[self.num_snorna:]
        
        # 预测所有关联（向量化计算）
        # 扩展维度以进行批量计算
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)
        
        # 拼接特征
        combined = torch.cat([snorna_expanded, disease_expanded], dim=2)
        combined = combined.view(-1, combined.size(-1))
        
        # 预测
        scores = self.predictor(combined)
        scores = scores.view(self.num_snorna, self.num_disease)
        
        return scores


class AssociationDataset(Dataset):
    """关联数据集"""
    
    def __init__(self, pos_samples, neg_samples):
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        
        # 合并样本
        self.samples = np.vstack([pos_samples, neg_samples])
        self.labels = np.hstack([
            np.ones(len(pos_samples)),
            np.zeros(len(neg_samples))
        ])
        
        # 打乱
        indices = np.random.permutation(len(self.samples))
        self.samples = self.samples[indices]
        self.labels = self.labels[indices]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class DataLoaderClass:
    """数据加载类"""
    
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        
    def load_all_data(self):
        """加载所有数据"""
        print("\n[步骤 1] 正在加载数据...")
        
        # 加载关联矩阵
        adj_df = pd.read_csv(f'{self.data_path}adj_index.csv', index_col=0)
        self.association_matrix = adj_df.values.astype(np.float32)
        self.snorna_names = adj_df.index.tolist()
        self.disease_names = adj_df.columns.tolist()
        
        # 加载相似度矩阵
        self.snorna_similarity = np.load(f'{self.data_path}GIPK-s.npy').astype(np.float32)
        self.disease_similarity = np.load(f'{self.data_path}GIPK-d.npy').astype(np.float32)
        
        # 归一化
        self.snorna_similarity = self._normalize_similarity(self.snorna_similarity)
        self.disease_similarity = self._normalize_similarity(self.disease_similarity)
        
        print(f"  ✓ snoRNA数量: {len(self.snorna_names)}")
        print(f"  ✓ Disease数量: {len(self.disease_names)}")
        print(f"  ✓ 已知关联数量: {int(self.association_matrix.sum())}")
        
        return self.association_matrix, self.snorna_similarity, self.disease_similarity
    
    def _normalize_similarity(self, sim_matrix):
        """归一化相似度矩阵"""
        np.fill_diagonal(sim_matrix, 1.0)
        sim_matrix = np.clip(sim_matrix, 0, 1)
        return sim_matrix


class HypergraphConstructor:
    """超图构造器"""
    
    def __init__(self, association_matrix, snorna_sim, disease_sim):
        self.association_matrix = association_matrix
        self.snorna_sim = snorna_sim
        self.disease_sim = disease_sim
        self.num_snorna = association_matrix.shape[0]
        self.num_disease = association_matrix.shape[1]
        
    def construct_hypergraph(self, k_snorna=10, k_disease=10):
        """构建超图"""
        print(f"\n[步骤 2] 正在构建超图 (k_snorna={k_snorna}, k_disease={k_disease})...")
        
        # 构建KNN图
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)
        
        # 构建超图关联矩阵
        num_nodes = self.num_snorna + self.num_disease
        num_hyperedges = self.num_snorna + self.num_disease
        
        H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)
        
        # snoRNA超边
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]
            for neighbor in neighbors:
                H[neighbor, i] = snorna_knn[i, neighbor]
        
        # disease超边
        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                H[self.num_snorna + neighbor, self.num_snorna + j] = disease_knn[j, neighbor]
        
        print(f"  ✓ 超图构建完成: {num_nodes} 节点, {num_hyperedges} 超边")
        
        return torch.FloatTensor(H).to(device)
    
    def _build_knn_graph(self, similarity_matrix, k):
        """构建KNN图"""
        n = similarity_matrix.shape[0]
        knn_graph = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            sim_scores = similarity_matrix[i].copy()
            sim_scores[i] = -1
            top_k_indices = np.argsort(sim_scores)[-k:]
            knn_graph[i, top_k_indices] = similarity_matrix[i, top_k_indices]
        
        # 对称化
        knn_graph = (knn_graph + knn_graph.T) / 2
        
        return knn_graph


class Trainer:
    """训练器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def train_epoch(self, H, train_pos, train_neg, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        
        # 前向传播
        predictions = self.model(H)
        
        # 计算损失
        loss = 0
        count = 0
        
        # 正样本
        for i, j in train_pos:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.ones(1).to(self.device))
            count += 1
        
        # 负样本
        for i, j in train_neg:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.zeros(1).to(self.device))
            count += 1
        
        loss = loss / count
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, H, test_pos, test_neg):
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(H)
        
        y_true = []
        y_scores = []
        
        # 正样本
        for i, j in test_pos:
            y_true.append(1)
            y_scores.append(predictions[i, j].cpu().item())
        
        # 负样本
        for i, j in test_neg:
            y_true.append(0)
            y_scores.append(predictions[i, j].cpu().item())
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # 计算指标
        auc_score = roc_auc_score(y_true, y_scores)
        aupr_score = average_precision_score(y_true, y_scores)
        
        return auc_score, aupr_score, y_true, y_scores


def prepare_samples(association_matrix, train_indices, test_indices):
    """准备训练和测试样本"""
    num_snorna, num_disease = association_matrix.shape
    
    # 获取所有正样本
    pos_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_samples.append((i, j))
    
    pos_samples = np.array(pos_samples)
    
    # 划分
    train_pos = pos_samples[train_indices]
    test_pos = pos_samples[test_indices]
    
    # 生成负样本
    neg_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 0:
                neg_samples.append((i, j))
    
    neg_samples = np.array(neg_samples)
    np.random.shuffle(neg_samples)
    
    train_neg = neg_samples[:len(train_pos)]
    test_neg = neg_samples[len(train_pos):len(train_pos) + len(test_pos)]
    
    return train_pos, train_neg, test_pos, test_neg


def cross_validation(association_matrix, snorna_sim, disease_sim, 
                     n_splits=5, epochs=100, lr=0.001, patience=20):
    """K折交叉验证"""
    print(f"\n[步骤 3] 开始 {n_splits} 折交叉验证...")
    print("="*80)
    
    # 构建超图
    hg_constructor = HypergraphConstructor(association_matrix, snorna_sim, disease_sim)
    H = hg_constructor.construct_hypergraph(k_snorna=10, k_disease=10)
    
    # 获取所有正样本
    pos_indices = []
    num_snorna, num_disease = association_matrix.shape
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_indices.append((i, j))
    
    num_pos = len(pos_indices)
    print(f"总正样本数: {num_pos}")
    
    # K折交叉验证
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_y_true = []
    all_y_scores = []
    all_fold_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(num_pos))):
        print(f"\n{'─'*80}")
        print(f"折 {fold + 1}/{n_splits}")
        print(f"{'─'*80}")
        
        # 准备样本
        train_pos, train_neg, test_pos, test_neg = prepare_samples(
            association_matrix, train_idx, test_idx
        )
        
        print(f"训练集 - 正样本: {len(train_pos)}, 负样本: {len(train_neg)}")
        print(f"测试集 - 正样本: {len(test_pos)}, 负样本: {len(test_neg)}")
        
        # 创建模型
        model = DeepHypergraphNN(
            num_snorna=num_snorna,
            num_disease=num_disease,
            snorna_sim=snorna_sim,
            disease_sim=disease_sim,
            hidden_dims=[256, 128, 64],  # 原始参数值hidden_dims=[256, 128, 64]
            num_heads=8,  # 原始参数值num_heads=8
            dropout=0.3  # 原始参数值dropout=0.3
        ).to(device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=False
        )
        criterion = nn.BCELoss()
        
        # 训练器
        trainer = Trainer(model, device)
        
        # 训练
        best_auc = 0
        patience_counter = 0
        
        progress_bar = tqdm(range(epochs), desc=f"训练折 {fold+1}")
        for epoch in progress_bar:
            loss = trainer.train_epoch(H, train_pos, train_neg, optimizer, criterion)
            
            # 每10个epoch评估一次
            if (epoch + 1) % 10 == 0:
                auc_score, aupr_score, _, _ = trainer.evaluate(H, test_pos, test_neg)
                
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'AUC': f'{auc_score:.4f}',
                    'AUPR': f'{aupr_score:.4f}'
                })
                
                scheduler.step(auc_score)
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n早停于 epoch {epoch+1}")
                    break
        
        # 最终评估
        auc_score, aupr_score, y_true, y_scores = trainer.evaluate(H, test_pos, test_neg)
        
        print(f"\n折 {fold + 1} 最终结果:")
        print(f"  AUC:  {auc_score:.4f}")
        print(f"  AUPR: {aupr_score:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'auc': auc_score,
            'aupr': aupr_score,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })
        
        all_y_true.extend(y_true)
        all_y_scores.extend(y_scores)
        all_fold_predictions.append({
            'y_true': y_true,
            'y_scores': y_scores
        })
        
        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算平均性能
    avg_auc = np.mean([r['auc'] for r in fold_results])
    std_auc = np.std([r['auc'] for r in fold_results])
    avg_aupr = np.mean([r['aupr'] for r in fold_results])
    std_aupr = np.std([r['aupr'] for r in fold_results])
    
    print(f"\n{'='*80}")
    print("交叉验证总结")
    print(f"{'='*80}")
    print(f"平均 AUC:  {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"平均 AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")
    print(f"{'='*80}")
    
    return fold_results, all_y_true, all_y_scores, all_fold_predictions


class ResultVisualizer:
    """结果可视化"""
    
    def __init__(self, output_dir='./outputs/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_all_results(self, fold_results, all_y_true, all_y_scores, fold_predictions):
        """生成所有可视化"""
        print(f"\n[步骤 4] 生成可视化图表...")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self._plot_fold_comparison(fold_results)
        self._plot_overall_roc(all_y_true, all_y_scores)
        self._plot_overall_pr(all_y_true, all_y_scores)
        self._plot_folds_roc(fold_predictions)
        self._plot_folds_pr(fold_predictions)
        self._plot_metrics_boxplot(fold_results)
        self._plot_metrics_heatmap(fold_results)
        self._plot_comprehensive_panel(fold_results, all_y_true, all_y_scores)
        
        print(f"  ✓ 所有图表已保存到: {self.output_dir}/")
    
    def _plot_fold_comparison(self, fold_results):
        """折间对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        # AUC
        bars1 = axes[0].bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
        axes[0].axhline(y=np.mean(aucs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(aucs):.4f}')
        axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
        axes[0].set_title('AUC Score across Folds (PyTorch Deep Learning)', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([min(aucs) - 0.05, 1.0])
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AUPR
        bars2 = axes[1].bar(folds, auprs, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5)
        axes[1].axhline(y=np.mean(auprs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(auprs):.4f}')
        axes[1].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('AUPR', fontsize=12, fontweight='bold')
        axes[1].set_title('AUPR Score across Folds (PyTorch Deep Learning)', 
                         fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([min(auprs) - 0.05, 1.0])
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_01_fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_01_fold_comparison.png")
    
    def _plot_overall_roc(self, y_true, y_scores):
        """整体ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Overall ROC Curve (PyTorch Deep Learning)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_02_overall_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_02_overall_roc_curve.png")
    
    def _plot_overall_pr(self, y_true, y_scores):
        """整体PR曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=3, label=f'PR curve (AUPR = {pr_auc:.4f})')
        baseline = sum(y_true) / len(y_true)
        plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
                label=f'Random (baseline={baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Overall Precision-Recall Curve (PyTorch Deep Learning)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_03_overall_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_03_overall_pr_curve.png")
    
    def _plot_folds_roc(self, fold_predictions):
        """所有折的ROC曲线"""
        plt.figure(figsize=(10, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(fold_predictions)))
        
        for i, fold_pred in enumerate(fold_predictions):
            fpr, tpr, _ = roc_curve(fold_pred['y_true'], fold_pred['y_scores'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'Fold {i+1} (AUC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves for All Folds (PyTorch Deep Learning)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_04_all_folds_roc.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_04_all_folds_roc.png")
    
    def _plot_folds_pr(self, fold_predictions):
        """所有折的PR曲线"""
        plt.figure(figsize=(10, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(fold_predictions)))
        
        for i, fold_pred in enumerate(fold_predictions):
            precision, recall, _ = precision_recall_curve(fold_pred['y_true'], 
                                                          fold_pred['y_scores'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, color=colors[i], 
                    label=f'Fold {i+1} (AUPR={pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curves for All Folds (PyTorch Deep Learning)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_05_all_folds_pr.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_05_all_folds_pr.png")
    
    def _plot_metrics_boxplot(self, fold_results):
        """性能箱线图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        bp1 = axes[0].boxplot([aucs], labels=['AUC'], patch_artist=True, widths=0.5)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('navy')
            patch.set_linewidth(2)
        
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('AUC Distribution (PyTorch Deep Learning)', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([min(aucs) - 0.05, 1.0])
        
        stats_text = f'Mean: {np.mean(aucs):.4f}\nStd: {np.std(aucs):.4f}\nMedian: {np.median(aucs):.4f}'
        axes[0].text(0.98, 0.02, stats_text, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        bp2 = axes[1].boxplot([auprs], labels=['AUPR'], patch_artist=True, widths=0.5)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_edgecolor('darkred')
            patch.set_linewidth(2)
        
        axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[1].set_title('AUPR Distribution (PyTorch Deep Learning)', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([min(auprs) - 0.05, 1.0])
        
        stats_text = f'Mean: {np.mean(auprs):.4f}\nStd: {np.std(auprs):.4f}\nMedian: {np.median(auprs):.4f}'
        axes[1].text(0.98, 0.02, stats_text, transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_06_metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_06_metrics_boxplot.png")
    
    def _plot_metrics_heatmap(self, fold_results):
        """性能热图"""
        data = np.array([[r['auc'], r['aupr']] for r in fold_results])
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(data.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        plt.xticks(range(len(fold_results)), [f"Fold {r['fold']}" for r in fold_results])
        plt.yticks([0, 1], ['AUC', 'AUPR'])
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
        
        for i in range(len(fold_results)):
            for j in range(2):
                plt.text(i, j, f'{data[i, j]:.3f}', ha="center", va="center", 
                        color="black", fontweight='bold', fontsize=11)
        
        plt.xlabel('Fold', fontsize=12, fontweight='bold')
        plt.title('Performance Metrics Heatmap (PyTorch Deep Learning)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/DL_07_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_07_metrics_heatmap.png")
    
    def _plot_comprehensive_panel(self, fold_results, y_true, y_scores):
        """综合面板图"""
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        # 1. 性能对比
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(folds))
        width = 0.35
        ax1.bar(x - width/2, aucs, width, label='AUC', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, auprs, width, label='AUPR', color='coral', alpha=0.8)
        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(A) Performance Comparison (PyTorch)', fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. ROC曲线
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')
        ax2.set_title('(B) ROC Curve', fontweight='bold', loc='left')
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)
        
        # 3. PR曲线
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color='blue', lw=3, label=f'AUPR = {pr_auc:.4f}')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('(C) Precision-Recall Curve', fontweight='bold', loc='left')
        ax3.legend(loc="lower left")
        ax3.grid(alpha=0.3)
        
        # 4. 统计表格
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Mean', 'Std', 'Min', 'Max'],
            ['AUC', f'{np.mean(aucs):.4f}', f'{np.std(aucs):.4f}', 
             f'{np.min(aucs):.4f}', f'{np.max(aucs):.4f}'],
            ['AUPR', f'{np.mean(auprs):.4f}', f'{np.std(auprs):.4f}', 
             f'{np.min(auprs):.4f}', f'{np.max(auprs):.4f}']
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, 3):
            for j in range(5):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
        
        ax4.set_title('(D) Statistical Summary', fontweight='bold', loc='left', pad=20)
        
        plt.savefig(f'{self.output_dir}/DL_08_comprehensive_panel.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: DL_08_comprehensive_panel.png")


def save_results(fold_results, output_dir='./outputs/'):
    """保存结果"""
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(f'{output_dir}/DL_fold_results.csv', index=False)
    
    aucs = [r['auc'] for r in fold_results]
    auprs = [r['aupr'] for r in fold_results]
    
    summary = {
        'Metric': ['AUC', 'AUPR'],
        'Mean': [np.mean(aucs), np.mean(auprs)],
        'Std': [np.std(aucs), np.std(auprs)],
        'Min': [np.min(aucs), np.min(auprs)],
        'Max': [np.max(aucs), np.max(auprs)],
        'Median': [np.median(aucs), np.median(auprs)]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f'{output_dir}/DL_summary_statistics.csv', index=False)
    
    print(f"\n[步骤 5] 结果已保存")
    print(f"  ✓ DL_fold_results.csv")
    print(f"  ✓ DL_summary_statistics.csv")


def main():
    """主函数"""
    print("\n开始PyTorch深度学习训练流程...\n")
    
    # 加载数据
    data_loader = DataLoaderClass()
    association_matrix, snorna_sim, disease_sim = data_loader.load_all_data()
    
    # 交叉验证
    fold_results, all_y_true, all_y_scores, fold_predictions = cross_validation(
        association_matrix=association_matrix,
        snorna_sim=snorna_sim,
        disease_sim=disease_sim,
        n_splits=5,
        epochs=100,
        lr=0.001,
        patience=20
    )
    
    # 可视化
    visualizer = ResultVisualizer()
    visualizer.plot_all_results(fold_results, all_y_true, all_y_scores, fold_predictions)
    
    # 保存结果
    save_results(fold_results)
    
    print("\n" + "="*80)
    print(" PyTorch深度学习训练完成！")
    print("="*80)
    print(f"\n所有结果已保存到: ./outputs/")
    print("\n生成的文件 (以 DL_ 开头):")
    print("  - DL_01_fold_comparison.png")
    print("  - DL_02_overall_roc_curve.png")
    print("  - DL_03_overall_pr_curve.png")
    print("  - DL_04_all_folds_roc.png")
    print("  - DL_05_all_folds_pr.png")
    print("  - DL_06_metrics_boxplot.png")
    print("  - DL_07_metrics_heatmap.png")
    print("  - DL_08_comprehensive_panel.png")
    print("  - DL_fold_results.csv")
    print("  - DL_summary_statistics.csv")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
