# Author:Hibari
# 2025年11月18日17时31分50秒
# syh19990131@gmail.com
# hypergraph.py
import numpy as np
import torch
from utils import device


# 超图构造器）
class HypergraphConstructor:
    """
    超图构造器

    创新点：同时使用GIPK相似性和关联矩阵构造超边
    """

    def __init__(self, association_matrix, snorna_sim, disease_sim):
        """
        初始化超图构造器

        参数:
            association_matrix: snoRNA-disease关联矩阵
            snorna_sim: snoRNA的GIPK相似性矩阵
            disease_sim: disease的GIPK相似性矩阵
        """
        self.association_matrix = association_matrix
        self.snorna_sim = snorna_sim
        self.disease_sim = disease_sim
        self.num_snorna = association_matrix.shape[0]
        self.num_disease = association_matrix.shape[1]

    def construct_hypergraph(self, k_snorna=15, k_disease=15,
                             use_association_edges=True,
                             association_weight=0.5):
        """
        构建超图关联矩阵（融合相似性和关联信息）

        参数:
            k_snorna: snoRNA的K近邻数量
            k_disease: disease的K近邻数量
            use_association_edges: 是否使用关联矩阵构造额外的超边
            association_weight: 关联超边的权重

        返回:
            H: 超图关联矩阵 [num_nodes, num_hyperedges]
        """
        print(f"\n[步骤 2] 构建改进的超图...")
        print(f"  - K近邻: snoRNA={k_snorna}, disease={k_disease}")
        print(f"  - 使用关联超边: {use_association_edges}")
        if use_association_edges:
            print(f"  - 关联超边权重: {association_weight}")

        # 构建 K 近邻图，基于GIPK相似性的超边
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)

        # 计算维度
        num_nodes = self.num_snorna + self.num_disease

        # 计算超边数量
        if use_association_edges:
            # 相似性超边 + 关联超边
            num_sim_edges = self.num_snorna + self.num_disease
            num_assoc_edges = int(self.association_matrix.sum())  # 已知关联数
            num_hyperedges = num_sim_edges + num_assoc_edges
        else:
            # 仅相似性超边
            num_hyperedges = self.num_snorna + self.num_disease

        # 初始化超图关联矩阵，超图H行代表节点（snoRNA 和 disease）。列代表超边，元素H[i,j]代表节点 i 在超边 j 中的权重
        H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)

        # 填充 snoRNA 相似性超边
        edge_idx = 0

        # 填充snoRNA相似性超边
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]  # 找到所有邻居
            for neighbor in neighbors:
                H[neighbor, edge_idx] = snorna_knn[i, neighbor]  # # 使用相似度作为权重
            edge_idx += 1

        # 填充disease相似性超边
        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                # disease 节点索引需要偏移 self.num_snorna
                H[self.num_snorna + neighbor, edge_idx] = disease_knn[j, neighbor]
            edge_idx += 1

        print(f"  ✓ 相似性超边构建完成: {edge_idx} 条超边")

        # 填充关联超边
        if use_association_edges:
            num_assoc_edges_added = 0

            # 为每个已知的snoRNA-disease关联创建一条超边
            for i in range(self.num_snorna):
                for j in range(self.num_disease):
                    if self.association_matrix[i, j] == 1:
                        # 创建一条连接snoRNA i和disease j的超边
                        H[i, edge_idx] = association_weight  # snoRNA节点
                        H[self.num_snorna + j, edge_idx] = association_weight  # disease节点
                        edge_idx += 1
                        num_assoc_edges_added += 1

            print(f"  ✓ 关联超边构建完成: {num_assoc_edges_added} 条超边")
            print(f"    （基于 {int(self.association_matrix.sum())} 个已知关联）")

        print(f"  ✓ 超图构建完成:")
        print(f"    - 总节点数: {num_nodes} ({self.num_snorna} snoRNA + {self.num_disease} disease)")
        print(f"    - 总超边数: {edge_idx}")

        return torch.FloatTensor(H).to(device)

    def _build_knn_graph(self, similarity_matrix, k):  # 从完整的相似性矩阵中提取 K 近邻图。
        """从相似度矩阵构建K近邻图"""
        n = similarity_matrix.shape[0]
        knn_graph = np.zeros((n, n), dtype=np.float32)  # 初始化空图

        for i in range(n):
            sim_scores = similarity_matrix[i].copy()  # 复制相似度分数
            sim_scores[i] = -1  # 排除自己
            top_k_indices = np.argsort(sim_scores)[-k:]  # # 找到 top-K 相似的节点
            knn_graph[i, top_k_indices] = similarity_matrix[i, top_k_indices]

        knn_graph = (knn_graph + knn_graph.T) / 2  # 对称化

        return knn_graph

    def construct_multi_hypergraph(self, k_snorna=15, k_disease=15,
                                   use_association_edges=True,
                                   association_weight=0.5):
        """
        构建三个超图：
          - H_all    : 原来代码里的“相似性 + 关联”超图（保持完全兼容）
          - H_sno    : 只包含 snoRNA KNN 相似性超边
          - H_disease: 只包含 disease KNN 相似性超边

        返回:
            H_all, H_sno, H_disease
        """
        # 先用原来的函数构建 H_all（包含相似性 + 关联信息）
        H_all = self.construct_hypergraph(
            k_snorna=k_snorna,
            k_disease=k_disease,
            use_association_edges=use_association_edges,
            association_weight=association_weight
        )

        # 基于当前折的 GIPK 相似性构建 KNN 图
        snorna_knn = self._build_knn_graph(self.snorna_sim, k_snorna)
        disease_knn = self._build_knn_graph(self.disease_sim, k_disease)

        num_nodes = self.num_snorna + self.num_disease

        # === H_sno：只用 snoRNA 相似性构造的超图 ===
        # 每个 snoRNA 节点对应一条超边，超边里是它的 K 近邻
        H_sno = np.zeros((num_nodes, self.num_snorna), dtype=np.float32)
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]
            for neighbor in neighbors:
                # 这里只在 snoRNA 节点之间连边，因此行索引是 snoRNA 的编号
                H_sno[neighbor, i] = snorna_knn[i, neighbor]

        # === H_disease：只用 disease 相似性构造的超图 ===
        H_dis = np.zeros((num_nodes, self.num_disease), dtype=np.float32)
        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                # disease 节点在整体节点中的索引要加上 self.num_snorna 偏移
                H_dis[self.num_snorna + neighbor, j] = disease_knn[j, neighbor]

        H_sno = torch.FloatTensor(H_sno).to(device)
        H_dis = torch.FloatTensor(H_dis).to(device)

        print(f"  ✓ 多超图构建完成: "
              f"H_all={tuple(H_all.shape)}, "
              f"H_sno={tuple(H_sno.shape)}, "
              f"H_dis={tuple(H_dis.shape)}")

        return H_all, H_sno, H_dis
