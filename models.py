# Author:Hibari
# 2025年11月18日17时33分01秒
# syh19990131@gmail.com
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):  # 初始化GAT层，创建可学习参数
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W矩阵 (F×F'): 特征变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # a向量 (2F'×1): 注意力参数向量
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU: 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # # 步骤1: 线性变换 (N×F) → (N×F')
        N = h.size()[0]

        # 步骤2: 构造注意力输入 - 为所有节点对创建拼接特征
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                             h.repeat(N, 1)], dim=1).view(N, -1,
                                                          2 * self.out_features)  # # 结果: a_input[i,j] = [h_i || h_j]
        # 步骤3: 计算注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 步骤4: 应用邻接矩阵掩码（只保留邻居的注意力）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 步骤5: Softmax归一化
        attention = F.softmax(attention, dim=1)
        # 步骤6: Dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 步骤7: 加权聚合
        h_prime = torch.matmul(attention, h)
        return h_prime


class MultiHeadGAT(nn.Module):
    """多头图注意力"""

    def __init__(self, in_features, out_features, num_heads=8,
                 dropout=0.3):  # 初始化多头图注意力层，创建多个并行的图注意力头。in_features: 输入特征维度;out_features: 输出特征维度;num_heads: 注意力头的数量（默认8个）;dropout: Dropout比率（默认0.3）
        super(MultiHeadGAT, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, dropout)
            for _ in range(num_heads)
        ])

    def forward(self, x, adj):
        """
        x: 输入节点特征矩阵，形状为 [N, in_features]，N为节点数
        adj: 邻接矩阵，形状为 [N, N]，表示图的连接关系
        """
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 并行计算+特征拼接
        return x


class EnhancedHypergraphConvolution(nn.Module):
    """超图卷积层"""

    def __init__(self, in_features, out_features, bias=True, dropout=0.3):  # 初始化超图卷积层的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        bias：是否使用偏置（默认True）
        dropout：Dropout比率（默认0.3）
        """
        super(EnhancedHypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 双路径线性变换层
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(in_features, out_features, bias=bias)

        # 批归一化层（稳定训练，加速收敛）
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Dropout层（防止过拟合）
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):  # 执行超图卷积的前向传播
        """
        超图卷积

        参数:
            X: 节点特征矩阵 [N, in_features]
            H: 超图关联矩阵 [N, M]
        """
        # 计算度矩阵
        D_v = torch.sum(H, dim=1, keepdim=True).clamp(min=1)
        D_e = torch.sum(H, dim=0, keepdim=True).clamp(min=1)

        # Path 1: 标准超图卷积
        # 归一化超图关联矩阵
        H_norm = H / torch.sqrt(D_v)
        H_norm = H_norm / torch.sqrt(D_e)
        # 超图卷积操作
        X1 = self.linear1(X)
        X1 = H_norm @ (H_norm.T @ X1)
        X1 = self.bn1(X1)
        X1 = self.dropout(X1)

        # Path 2: 跳跃连接,保留原始特征信息，类似于ResNet中的残差连接
        X2 = self.linear2(X)  # 直接线性变换
        X2 = self.bn2(X2)  # 批归一化
        X2 = self.dropout(X2)  # Dropout

        return F.elu(X1 + X2)


class DualAttentionModule(nn.Module):
    """双重注意力模块"""

    def __init__(self, dim, num_heads=8, dropout=0.3):  # 初始化双重注意力模块的所有组件
        """
        dim：特征维度
        num_heads：多头注意力的头数（默认8）
        dropout：Dropout比率（默认0.3）
        """
        super(DualAttentionModule, self).__init__()

        self.spatial_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                       batch_first=True)  # dim:特征维度,num_head:注意力头数,dropout=dropout,batch_first=True  输入格式：[batch, seq, feature]

        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),  # 降维：压缩信息
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 正则化
            nn.Linear(dim // 4, dim),  # 升维：恢复维度
            nn.Sigmoid()  # 输出0-1权重
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)  # 空间注意力后的归一化
        self.norm2 = nn.LayerNorm(dim)  # 通道注意力后的归一化

    def forward(self, x):
        # x：输入特征矩阵，形状为 [N, dim]，其中N是节点数量
        # 空间注意力
        x_unsqueezed = x.unsqueeze(0)  # 增加batch维度：[N, dim] → [1, N, dim]
        attn_out, _ = self.spatial_attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)  # 多头自注意力
        x = self.norm1(x + attn_out.squeeze(0))  # 残差连接 + 层归一化

        # 通道注意力
        channel_weights = self.channel_attention(x.mean(dim=0, keepdim=True))  # 计算通道权重：对所有节点取平均，得到全局统计信息
        x = self.norm2(x * channel_weights)  # 应用通道权重 + 残差连接 + 层归一化

        return x


class AdvancedHypergraphBlock(nn.Module):
    """超图块"""

    def __init__(self, in_features, out_features, num_heads=8, dropout=0.3):  # 初始化高级超图块的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        num_heads：多头注意力的头数（默认8）
        dropout：Dropout比率（默认0.3）
        """
        super(AdvancedHypergraphBlock, self).__init__()

        self.hgc = EnhancedHypergraphConvolution(in_features, out_features, dropout=dropout)  # 增强超图卷积层
        self.dual_attention = DualAttentionModule(out_features, num_heads, dropout)  # 双重注意力模块

        self.ffn = nn.Sequential(  # 前馈神经网络（FFN）
            nn.Linear(out_features, out_features * 4),  # 扩展4倍
            nn.GELU(),  # 平滑激活函数
            nn.Dropout(dropout),
            nn.Linear(out_features * 4, out_features),  # 压缩回原维度
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(out_features)  # 超图卷积后
        self.norm2 = nn.LayerNorm(out_features)  # 双重注意力后

        self.residual_proj = nn.Linear(in_features,
                                       out_features) if in_features != out_features else nn.Identity()  # 处理维度不匹配的残差连接

    def forward(self, X, H):  # 执行高级超图块的前向传播
        identity = self.residual_proj(X)  # 保存残差基准

        # 超图卷积 + 残差 + 归一化
        X = self.hgc(X, H)
        # 残差连接 + 层归一化
        X = self.norm1(X + identity)

        # 双重注意力
        X_att = self.dual_attention(X)
        # 残差连接 + 层归一化
        X = self.norm2(X + X_att)

        # 前馈网络 + 残差
        X = X + self.ffn(X)

        return X


class FeatureEnhancementModule(nn.Module):
    """特征增强模块"""

    def __init__(self, in_features, out_features, dropout=0.3):  # 初始化特征增强模块的所有组件
        """
        in_features：输入特征维度
        out_features：输出特征维度
        dropout：Dropout比率（默认0.3）
        """
        super(FeatureEnhancementModule, self).__init__()

        self.enhance = nn.Sequential(
            # 扩展
            nn.Linear(in_features, out_features * 2),  # 扩展到2倍
            nn.BatchNorm1d(out_features * 2),  # 批归一化
            nn.ELU(),  # ELU激活
            nn.Dropout(dropout),  # Dropout正则化

            # 压缩
            nn.Linear(out_features * 2, out_features),  # 压缩到目标维度
            nn.BatchNorm1d(out_features),  # 批归一化
            nn.ELU(),  # ELU激活
            nn.Dropout(dropout)  # Dropout正则化
        )

    def forward(self, x):  # 执行特征增强的前向传播
        return self.enhance(x)


class DeepHypergraphNN(nn.Module):
    """深度超图神经网络"""

    def __init__(self, num_snorna, num_disease, snorna_sim, disease_sim,
                 hidden_dims=[512, 384, 256, 128, 64], num_heads=8, dropout=0.2):  # 初始化深度超图神经网络的所有组件
        """
        num_snorna：snoRNA的数量
        num_disease：disease的数量
        snorna_sim：snoRNA的GIPK相似性矩阵
        disease_sim：disease的GIPK相似性矩阵
        hidden_dims：隐藏层维度列表（默认[512, 384, 256, 128, 64]）
        num_heads：注意力头数（默认8）
        dropout：Dropout比率（默认0.2）
        """
        super(DeepHypergraphNN, self).__init__()

        self.num_snorna = num_snorna
        self.num_disease = num_disease

        # 可学习的特征嵌入
        self.snorna_features = nn.Parameter(torch.FloatTensor(snorna_sim), requires_grad=True)
        self.disease_features = nn.Parameter(torch.FloatTensor(disease_sim), requires_grad=True)

        # 特征增强
        self.snorna_enhance = FeatureEnhancementModule(num_snorna, hidden_dims[0], dropout)
        self.disease_enhance = FeatureEnhancementModule(num_disease, hidden_dims[0], dropout)

        # 多尺度特征提取
        branch_dims = [hidden_dims[0] // 4, hidden_dims[0] // 4, hidden_dims[0] // 2]

        # snoRNA多尺度分支
        self.snorna_multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_snorna, dim),
                nn.BatchNorm1d(dim),
                nn.ELU()
            ) for dim in branch_dims
        ])

        # disease多尺度分支
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
                AdvancedHypergraphBlock(dims[i], dims[i + 1], num_heads, dropout)
            )

        # 全局池化注意力
        self.global_attention = DualAttentionModule(hidden_dims[-1], num_heads, dropout)

        # 预测头
        final_dim = hidden_dims[-1]
        self.predictor = nn.Sequential(
            # 第1层：扩展
            nn.Linear(final_dim * 2, final_dim * 2),
            nn.BatchNorm1d(final_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第2层：压缩
            nn.Linear(final_dim * 2, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第3层：进一步压缩
            nn.Linear(final_dim, final_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            # 第4层：输出层
            nn.Linear(final_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):  # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 线性层（Linear）
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H, drop_edge_rate=0.0):  # 执行完整的前向传播
        # DropEdge数据增强
        if self.training and drop_edge_rate > 0:
            H = self._drop_edge(H, drop_edge_rate)

        # 多尺度特征提取
        # 三个分支并行处理
        snorna_features_multi = [scale(self.snorna_features) for scale in self.snorna_multi_scale]
        disease_features_multi = [scale(self.disease_features) for scale in self.disease_multi_scale]

        # 拼接多尺度特征
        snorna_feat = torch.cat(snorna_features_multi, dim=1)
        disease_feat = torch.cat(disease_features_multi, dim=1)

        # 特征增强与融合
        snorna_feat = snorna_feat + self.snorna_enhance(self.snorna_features)  # [num_snorna, 512]
        disease_feat = disease_feat + self.disease_enhance(self.disease_features)  # [num_disease, 512]

        # 拼接所有节点特征
        X = torch.cat([snorna_feat, disease_feat], dim=0)

        # 通过超图卷积块
        for hg_block in self.hg_blocks:
            X = hg_block(X, H)

        # 全局注意力
        X = self.global_attention(X)

        # 分离特征
        snorna_embed = X[:self.num_snorna]  # [361, 64]
        disease_embed = X[self.num_snorna:]  # [780, 64]

        # 预测所有关联
        # 扩展维度以计算所有配对
        snorna_expanded = snorna_embed.unsqueeze(1).expand(-1, self.num_disease, -1)
        disease_expanded = disease_embed.unsqueeze(0).expand(self.num_snorna, -1, -1)

        # 拼接配对特征
        combined = torch.cat([snorna_expanded, disease_expanded], dim=2)
        # 展平为二维矩阵
        combined = combined.view(-1, combined.size(-1))

        # 预测关联分数
        scores = self.predictor(combined)  # [281580, 128] → [281580, 1]
        scores = scores.view(self.num_snorna, self.num_disease)  # [361, 780]

        return scores

    def _drop_edge(self, H, rate):  # DropEdge数据增强
        """DropEdge数据增强"""
        mask = torch.rand_like(H) > rate
        return H * mask.float()
