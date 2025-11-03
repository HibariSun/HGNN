"""
高级优化版超图神经网络 - 目标: AUC>0.9659, AUPR>0.9522
包含多项先进技术的优化实现
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
import math
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):  # 设置所有随机种子以确保实验可重复性
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*80)
print(" 高级优化版超图神经网络 - snoRNA-Disease关联预测")
print("="*80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = pred.clamp(min=1e-7, max=1-1e-7)
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        
        # 注意力机制
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return h_prime


class MultiHeadGAT(nn.Module):
    """多头图注意力"""
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, dropout)
            for _ in range(num_heads)
        ])
        
    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return x


class EnhancedHypergraphConvolution(nn.Module):
    """增强版超图卷积层"""
    
    def __init__(self, in_features, out_features, bias=True, dropout=0.3):
        super(EnhancedHypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(in_features, out_features, bias=bias)
        
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, H):
        # 计算度矩阵
        D_v = torch.sum(H, dim=1, keepdim=True).clamp(min=1)
        D_e = torch.sum(H, dim=0, keepdim=True).clamp(min=1)
        
        # 双路径聚合
        # Path 1: 标准超图卷积
        H_norm = H / torch.sqrt(D_v)
        H_norm = H_norm / torch.sqrt(D_e)
        X1 = self.linear1(X)
        X1 = H_norm @ (H_norm.T @ X1)
        X1 = self.bn1(X1)
        X1 = self.dropout(X1)
        
        # Path 2: 跳跃连接
        X2 = self.linear2(X)
        X2 = self.bn2(X2)
        X2 = self.dropout(X2)
        
        # 融合
        return F.elu(X1 + X2)


class DualAttentionModule(nn.Module):
    """双重注意力模块"""
    
    def __init__(self, dim, num_heads=8, dropout=0.3):
        super(DualAttentionModule, self).__init__()
        
        # 空间注意力
        self.spatial_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 空间注意力
        x_unsqueezed = x.unsqueeze(0)
        attn_out, _ = self.spatial_attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = self.norm1(x + attn_out.squeeze(0))
        
        # 通道注意力
        channel_weights = self.channel_attention(x.mean(dim=0, keepdim=True))
        x = self.norm2(x * channel_weights)
        
        return x


class AdvancedHypergraphBlock(nn.Module):
    """高级超图块"""
    
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):
        super(AdvancedHypergraphBlock, self).__init__()
        
        # 超图卷积
        self.hgc = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout)
        
        # 双重注意力
        self.dual_attention = DualAttentionModule(out_features, num_heads, dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        # 残差投影
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, X, H):
        identity = self.residual_proj(X)
        
        # 超图卷积
        X = self.hgc(X, H)
        X = self.norm1(X + identity)
        
        # 双重注意力
        X_att = self.dual_attention(X)
        X = self.norm2(X + X_att)
        
        # 前馈网络
        X = X + self.ffn(X)
        
        return X


class FeatureEnhancementModule(nn.Module):
    """特征增强模块"""
    
    def __init__(self, in_features, out_features, dropout=0.3):
        super(FeatureEnhancementModule, self).__init__()
        
        self.enhance = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.BatchNorm1d(out_features * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.enhance(x)


class AdvancedDeepHypergraphNN(nn.Module):
    """高级深度超图神经网络"""
    
    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim,
                 hidden_dims=[512, 384, 256, 128, 64], num_heads=8, dropout=0.2):
        super(AdvancedDeepHypergraphNN, self).__init__()
        
        self.num_snorna = num_snorna
        self.num_disease = num_disease
        
        # 可学习的特征嵌入
        self.snorna_features = nn.Parameter(torch.FloatTensor(snorna_sim), requires_grad=True)
        self.disease_features = nn.Parameter(torch.FloatTensor(disease_sim), requires_grad=True)
        
        # 特征增强
        self.snorna_enhance = FeatureEnhancementModule(num_snorna, hidden_dims[0], dropout)
        self.disease_enhance = FeatureEnhancementModule(num_disease, hidden_dims[0], dropout)
        
        # 多尺度特征提取 - 修复维度匹配问题
        # 使用不同的分支维度，确保总和等于hidden_dims[0]
        branch_dims = [hidden_dims[0] // 4, hidden_dims[0] // 4, hidden_dims[0] // 2]
        
        self.snorna_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_snorna, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])
        
        self.disease_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_disease, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])
        
        # 超图卷积块
        self.hg_blocks = nn.ModuleList()
        dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(hidden_dims)):
            self.hg_blocks.append(
                AdvancedHypergraphBlock(dims[i], dims[i+1], num_heads, dropout)
            )
        
        # 全局池化注意力
        self.global_attention = DualAttentionModule(hidden_dims[-1], num_heads, dropout)
        
        # 预测头
        final_dim = hidden_dims[-1]
        self.predictor = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim * 2),
            nn.BatchNorm1d(final_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, H, drop_edge_rate=0.0):
        # DropEdge数据增强
        if self.training and drop_edge_rate > 0:
            H = self._drop_edge(H, drop_edge_rate)
        
        # 多尺度特征提取
        snorna_features_multi = [scale(self.snorna_features) for scale in self.snorna_multi_scale]
        disease_features_multi = [scale(self.disease_features) for scale in self.disease_multi_scale]
        
        snorna_feat = torch.cat(snorna_features_multi, dim=1)
        disease_feat = torch.cat(disease_features_multi, dim=1)
        
        # 特征增强
        snorna_feat = snorna_feat + self.snorna_enhance(self.snorna_features)
        disease_feat = disease_feat + self.disease_enhance(self.disease_features)
        
        # 拼接所有节点特征
        X = torch.cat([snorna_feat, disease_feat], dim=0)
        
        # 通过超图卷积块
        for hg_block in self.hg_blocks:
            X = hg_block(X, H)
        
        # 全局注意力
        X = self.global_attention(X)
        
        # 分离特征
        snorna_embed = X[:self.num_snorna]
        disease_embed = X[self.num_snorna:]
        
        # 预测所有关联
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)
        
        combined = torch.cat([snorna_expanded, disease_expanded], dim=2)
        combined = combined.view(-1, combined.size(-1))
        
        scores = self.predictor(combined)
        scores = scores.view(self.num_snorna, self.num_disease)
        
        return scores
    
    def _drop_edge(self, H, rate):
        """DropEdge数据增强"""
        mask = torch.rand_like(H) > rate
        return H * mask.float()


class WarmupCosineScheduler:
    """Warmup + 余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class DataLoaderClass:
    """数据加载类"""
    
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        
    def load_all_data(self):
        print("\n[步骤 1] 正在加载数据...")
        
        adj_df = pd.read_csv(f'{self.data_path}adj_index.csv', index_col=0)
        self.association_matrix = adj_df.values.astype(np.float32)
        
        self.snorna_similarity = np.load(f'{self.data_path}GIPK-s.npy').astype(np.float32)
        self.disease_similarity = np.load(f'{self.data_path}GIPK-d.npy').astype(np.float32)
        
        self.snorna_similarity = self._normalize_similarity(self.snorna_similarity)
        self.disease_similarity = self._normalize_similarity(self.disease_similarity)
        
        print(f"  ✓ snoRNA数量: {self.association_matrix.shape[0]}")
        print(f"  ✓ Disease数量: {self.association_matrix.shape[1]}")
        print(f"  ✓ 已知关联数量: {int(self.association_matrix.sum())}")
        
        return self.association_matrix, self.snorna_similarity, self.disease_similarity
    
    def _normalize_similarity(self, sim_matrix):
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
        
    def construct_hypergraph(self, k_snorna=15, k_disease=15):
        print(f"\n[步骤 2] 正在构建超图 (k_snorna={k_snorna}, k_disease={k_disease})...")
        
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)
        
        num_nodes = self.num_snorna + self.num_disease
        num_hyperedges = self.num_snorna + self.num_disease
        
        H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)
        
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]
            for neighbor in neighbors:
                H[neighbor, i] = snorna_knn[i, neighbor]
        
        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                H[self.num_snorna + neighbor, self.num_snorna + j] = disease_knn[j, neighbor]
        
        print(f"  ✓ 超图构建完成: {num_nodes} 节点, {num_hyperedges} 超边")
        
        return torch.FloatTensor(H).to(device)
    
    def _build_knn_graph(self, similarity_matrix, k):
        n = similarity_matrix.shape[0]
        knn_graph = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            sim_scores = similarity_matrix[i].copy()
            sim_scores[i] = -1
            top_k_indices = np.argsort(sim_scores)[-k:]
            knn_graph[i, top_k_indices] = similarity_matrix[i, top_k_indices]
        
        knn_graph = (knn_graph + knn_graph.T) / 2
        
        return knn_graph


def prepare_samples(association_matrix, train_indices, test_indices):
    num_snorna, num_disease = association_matrix.shape
    
    pos_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_samples.append((i, j))
    
    pos_samples = np.array(pos_samples)
    
    train_pos = pos_samples[train_indices]
    test_pos = pos_samples[test_indices]
    
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


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, model, device, label_smoothing=0.1):
        self.model = model
        self.device = device
        self.label_smoothing = label_smoothing
        
    def train_epoch(self, H, train_pos, train_neg, optimizer, criterion, drop_edge_rate=0.1):
        self.model.train()
        
        predictions = self.model(H, drop_edge_rate=drop_edge_rate)
        
        loss = 0
        count = 0
        
        # 标签平滑
        pos_label = 1.0 - self.label_smoothing
        neg_label = self.label_smoothing
        
        for i, j in train_pos:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.tensor([pos_label]).to(self.device))
            count += 1
        
        for i, j in train_neg:
            loss += criterion(predictions[i, j].unsqueeze(0), 
                            torch.tensor([neg_label]).to(self.device))
            count += 1
        
        loss = loss / count
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, H, test_pos, test_neg):
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(H)
        
        y_true = []
        y_scores = []
        
        for i, j in test_pos:
            y_true.append(1)
            y_scores.append(predictions[i, j].cpu().item())
        
        for i, j in test_neg:
            y_true.append(0)
            y_scores.append(predictions[i, j].cpu().item())
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        auc_score = roc_auc_score(y_true, y_scores)
        aupr_score = average_precision_score(y_true, y_scores)
        
        return auc_score, aupr_score, y_true, y_scores


def cross_validation(association_matrix, snorna_sim, disease_sim, 
                     n_splits=5, epochs=200, lr=0.0005, patience=30):
    print(f"\n[步骤 3] 开始 {n_splits} 折交叉验证...")
    print("="*80)
    
    hg_constructor = HypergraphConstructor(association_matrix, snorna_sim, disease_sim)
    H = hg_constructor.construct_hypergraph(k_snorna=15, k_disease=15)
    
    pos_indices = []
    num_snorna, num_disease = association_matrix.shape
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_indices.append((i, j))
    
    num_pos = len(pos_indices)
    print(f"总正样本数: {num_pos}")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    all_y_true = []
    all_y_scores = []
    all_fold_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(range(num_pos))):
        print(f"\n{'─'*80}")
        print(f"折 {fold + 1}/{n_splits}")
        print(f"{'─'*80}")
        
        train_pos, train_neg, test_pos, test_neg = prepare_samples(
            association_matrix, train_idx, test_idx
        )
        
        print(f"训练集 - 正样本: {len(train_pos)}, 负样本: {len(train_neg)}")
        print(f"测试集 - 正样本: {len(test_pos)}, 负样本: {len(test_neg)}")
        
        # 创建模型
        model = AdvancedDeepHypergraphNN(
            num_snorna=num_snorna,
            num_disease=num_disease,
            snorna_sim=snorna_sim,
            disease_sim=disease_sim,
            hidden_dims=[512, 384, 256, 128, 64],
            num_heads=8,
            dropout=0.2
        ).to(device)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        trainer = AdvancedTrainer(model, device, label_smoothing=0.05)
        
        best_auc = 0
        best_aupr = 0
        patience_counter = 0
        
        progress_bar = tqdm(range(epochs), desc=f"训练折 {fold+1}")
        for epoch in progress_bar:
            # 动态DropEdge
            drop_edge_rate = 0.1 * (1 - epoch / epochs)
            
            loss = trainer.train_epoch(H, train_pos, train_neg, optimizer, criterion, drop_edge_rate)
            scheduler.step(epoch)
            
            if (epoch + 1) % 5 == 0:
                auc_score, aupr_score, _, _ = trainer.evaluate(H, test_pos, test_neg)
                
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'AUC': f'{auc_score:.4f}',
                    'AUPR': f'{aupr_score:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_aupr = aupr_score
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
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    
    def __init__(self, output_dir='./outputs_advanced/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_all_results(self, fold_results, all_y_true, all_y_scores, fold_predictions):
        print(f"\n[步骤 4] 生成可视化图表...")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self._plot_fold_comparison(fold_results)
        self._plot_overall_roc(all_y_true, all_y_scores)
        self._plot_overall_pr(all_y_true, all_y_scores)
        self._plot_comprehensive_panel(fold_results, all_y_true, all_y_scores)
        
        print(f"  ✓ 所有图表已保存到: {self.output_dir}/")
    
    def _plot_fold_comparison(self, fold_results):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        bars1 = axes[0].bar(folds, aucs, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
        axes[0].axhline(y=np.mean(aucs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(aucs):.4f}')
        axes[0].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('AUC', fontsize=12, fontweight='bold')
        axes[0].set_title('AUC Score across Folds (Advanced)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        bars2 = axes[1].bar(folds, auprs, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5)
        axes[1].axhline(y=np.mean(auprs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(auprs):.4f}')
        axes[1].set_xlabel('Fold', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('AUPR', fontsize=12, fontweight='bold')
        axes[1].set_title('AUPR Score across Folds (Advanced)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ADV_01_fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: ADV_01_fold_comparison.png")
    
    def _plot_overall_roc(self, y_true, y_scores):
        plt.figure(figsize=(10, 8))
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Overall ROC Curve (Advanced Model)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/ADV_02_overall_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: ADV_02_overall_roc_curve.png")
    
    def _plot_overall_pr(self, y_true, y_scores):
        plt.figure(figsize=(10, 8))
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='blue', lw=3, 
                label=f'PR curve (AUPR = {pr_auc:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Overall Precision-Recall Curve (Advanced Model)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/ADV_03_overall_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: ADV_03_overall_pr_curve.png")
    
    def _plot_comprehensive_panel(self, fold_results, y_true, y_scores):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        folds = [r['fold'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        
        # 性能对比
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(folds))
        width = 0.35
        ax1.bar(x - width/2, aucs, width, label='AUC', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, auprs, width, label='AUPR', color='coral', alpha=0.8)
        ax1.set_xlabel('Fold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(A) Performance Comparison', fontweight='bold', loc='left')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # ROC曲线
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
        
        # PR曲线
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color='blue', lw=3, label=f'AUPR = {pr_auc:.4f}')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('(C) Precision-Recall Curve', fontweight='bold', loc='left')
        ax3.legend(loc="lower left")
        ax3.grid(alpha=0.3)
        
        # 统计表格
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
        
        ax4.set_title('(D) Statistical Summary', fontweight='bold', loc='left', pad=20)
        
        plt.savefig(f'{self.output_dir}/ADV_04_comprehensive_panel.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: ADV_04_comprehensive_panel.png")


def save_results(fold_results, output_dir='./outputs_advanced'):
    os.makedirs(output_dir, exist_ok=True)
    
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(f'{output_dir}/ADV_fold_results.csv', index=False)
    
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
    df_summary.to_csv(f'{output_dir}/ADV_summary_statistics.csv', index=False)
    
    print(f"\n[步骤 5] 结果已保存到: {output_dir}/")


def main():
    print("\n开始高级优化版训练流程...\n")
    
    data_loader = DataLoaderClass()
    association_matrix, snorna_sim, disease_sim = data_loader.load_all_data()
    
    fold_results, all_y_true, all_y_scores, fold_predictions = cross_validation(
        association_matrix=association_matrix,
        snorna_sim=snorna_sim,
        disease_sim=disease_sim,
        n_splits=5,
        epochs=200,
        lr=0.0005,
        patience=30
    )
    
    visualizer = ResultVisualizer()
    visualizer.plot_all_results(fold_results, all_y_true, all_y_scores, fold_predictions)
    
    save_results(fold_results)
    
    print("\n" + "="*80)
    print(" 高级优化版训练完成！")
    print("="*80)
    
    # 检查是否达到目标
    avg_auc = np.mean([r['auc'] for r in fold_results])
    avg_aupr = np.mean([r['aupr'] for r in fold_results])
    
    print(f"\n📊 性能评估:")
    print(f"  当前 AUC:  {avg_auc:.4f} (目标: > 0.9659)")
    print(f"  当前 AUPR: {avg_aupr:.4f} (目标: > 0.9522)")
    
    if avg_auc > 0.9659 and avg_aupr > 0.9522:
        print("\n🎉 恭喜！已达到目标性能！")
    else:
        print("\n💡 建议进一步优化:")
        if avg_auc <= 0.9659:
            print("  - 增加训练轮数 (epochs=300)")
            print("  - 减小学习率 (lr=0.0003)")
        if avg_aupr <= 0.9522:
            print("  - 调整Focal Loss参数 (alpha=0.8)")
            print("  - 增加隐藏层维度")


if __name__ == "__main__":
    main()
