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
        构建三种超图：
            1) 原有的混合超图（相似性 + 可选关联超边）
            2) 仅基于 snoRNA 的高阶超图（行特征）
            3) 仅基于 disease 的高阶超图（列特征）

        返回: (H_main, H_snorna, H_disease)
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

        # 计算超边数量（主超图）
        if use_association_edges:
            num_sim_edges = self.num_snorna + self.num_disease
            num_assoc_edges = int(self.association_matrix.sum())
            num_hyperedges = num_sim_edges + num_assoc_edges
        else:
            num_hyperedges = self.num_snorna + self.num_disease

        # 主超图 H_main
        H_main = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)

        edge_idx = 0

        for i in range(self.num_snorna):
            neighbors = np.where(snorna_knn[i] > 0)[0]
            for neighbor in neighbors:
                H_main[neighbor, edge_idx] = snorna_knn[i, neighbor]
            edge_idx += 1

        for j in range(self.num_disease):
            neighbors = np.where(disease_knn[j] > 0)[0]
            for neighbor in neighbors:
                H_main[self.num_snorna + neighbor, edge_idx] = disease_knn[j, neighbor]
            edge_idx += 1

        print(f"  ✓ 相似性超边构建完成: {edge_idx} 条超边")

        if use_association_edges:
            num_assoc_edges_added = 0
            for i in range(self.num_snorna):
                for j in range(self.num_disease):
                    if self.association_matrix[i, j] == 1:
                        H_main[i, edge_idx] = association_weight
                        H_main[self.num_snorna + j, edge_idx] = association_weight
                        edge_idx += 1
                        num_assoc_edges_added += 1

            print(f"  ✓ 关联超边构建完成: {num_assoc_edges_added} 条超边")
            print(f"    （基于 {int(self.association_matrix.sum())} 个已知关联）")

        print(f"  ✓ 主超图构建完成: 总超边数 {edge_idx}")

        # snoRNA 高阶超图（仅包含 snoRNA 节点的行特征相似性）
        snorna_feature_sim = self._build_feature_similarity(self.association_matrix)
        snorna_feature_knn = self._build_knn_graph(snorna_feature_sim, k_snorna)
        H_snorna = np.zeros((num_nodes, self.num_snorna), dtype=np.float32)
        edge_idx_sno = 0
        for i in range(self.num_snorna):
            neighbors = np.where(snorna_feature_knn[i] > 0)[0]
            for neighbor in neighbors:
                H_snorna[neighbor, edge_idx_sno] = snorna_feature_knn[i, neighbor]
            edge_idx_sno += 1

        # disease 高阶超图（仅包含 disease 节点的列特征相似性）
        disease_feature_sim = self._build_feature_similarity(self.association_matrix.T)
        disease_feature_knn = self._build_knn_graph(disease_feature_sim, k_disease)
        H_disease = np.zeros((num_nodes, self.num_disease), dtype=np.float32)
        edge_idx_dis = 0
        for j in range(self.num_disease):
            neighbors = np.where(disease_feature_knn[j] > 0)[0]
            for neighbor in neighbors:
                H_disease[self.num_snorna + neighbor, edge_idx_dis] = disease_feature_knn[j, neighbor]
            edge_idx_dis += 1

        print(f"  ✓ 额外超图构建完成: snoRNA {edge_idx_sno} 条, disease {edge_idx_dis} 条")

        return (
            torch.FloatTensor(H_main).to(device),
            torch.FloatTensor(H_snorna).to(device),
            torch.FloatTensor(H_disease).to(device),
        )

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

    def _build_feature_similarity(self, features):
        """基于行/列特征（adj 行/列）计算余弦相似度"""
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        normalized = features / norm
        similarity = normalized @ normalized.T
        similarity = np.clip(similarity, 0, 1)
        np.fill_diagonal(similarity, 1.0)
        return similarity
