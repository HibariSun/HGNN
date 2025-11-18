# Author:Hibari
# 2025年11月18日17时35分33秒
# syh19990131@gmail.com

# data_utils.py
import os
import numpy as np
import pandas as pd

from gipk import GIPKCalculator


# 数据加载类
class DataLoader:
    """改进的数据加载类 - 自己计算GIPK并融合外部相似性
       支持 snoRNA / disease 使用不同的 α，并做网格搜索时复用 GIPK 计算结果
    """

    def __init__(self, data_path='./data/',
                 alpha_snorna=None, alpha_disease=None):
        """
        alpha_snorna: snoRNA 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        alpha_disease: disease 相似性中 GIPK 所占比例 (0~1)，网格搜索时可覆盖
        """
        self.data_path = data_path
        self.alpha_snorna = alpha_snorna
        self.alpha_disease = alpha_disease

        # 缓存基础矩阵，避免在网格搜索时重复计算 GIPK
        self._association_matrix = None
        self._snorna_gipk = None
        self._disease_gipk = None
        self._sno_external = None
        self._disease_external = None
        self._external_available = False

    def load_all_data(self, alpha_snorna=None, alpha_disease=None):
        """
        加载数据并计算GIPK, 同时与外部相似性矩阵做 SWF 融合

        alpha_snorna / alpha_disease:
            - 如果为 None，则使用 __init__ 里传入的默认值
            - 如果仍为 None，则退回到“自动权重模式”（按均值算权重）

        返回:
            association_matrix: 关联矩阵
            snorna_sim_fused: 融合后的 snoRNA 相似性矩阵
            disease_sim_fused: 融合后的 disease 相似性矩阵
        """
        print("\n[步骤 1] 加载数据并计算相似性...")

        # ========== 第一次调用时，真正加载 & 计算 ==========
        if self._association_matrix is None:
            # 1. 加载关联矩阵
            adj_df = pd.read_csv(f"{self.data_path}adj_index.csv", index_col=0)
            self._association_matrix = adj_df.values.astype(np.float32)

            print(f"  ✓ 关联矩阵加载完成:")
            print(f"    - snoRNA数量: {self._association_matrix.shape[0]}")
            print(f"    - Disease数量: {self._association_matrix.shape[1]}")
            print(f"    - 已知关联数量: {int(self._association_matrix.sum())}")

            # 2. 基于关联矩阵计算 GIPK 相似性
            gipk_calculator = GIPKCalculator(self._association_matrix, gamma_ratio=1.0)
            snorna_gipk = gipk_calculator.compute_gipk_snorna()
            disease_gipk = gipk_calculator.compute_gipk_disease()

            self._snorna_gipk = self._normalize_similarity(snorna_gipk)
            self._disease_gipk = self._normalize_similarity(disease_gipk)

            print("\n  ✓ GIPK 计算和归一化完成")

            # 3. 加载外部相似性矩阵
            sno_ext_path = os.path.join(self.data_path, "sno_p2p_smith.csv")
            dis_ext_path = os.path.join(self.data_path, "disease_similarity.csv")

            self._external_available = os.path.exists(sno_ext_path) and os.path.exists(dis_ext_path)
            if not self._external_available:
                print("\n[警告] 未找到外部相似性文件 sno_p2p_smith.csv 或 disease_similarity.csv")
                print("       将仅使用 GIPK 相似性。要启用 SWF 融合，请将两个文件放到 data/ 目录下。")
            else:
                print("\n  ✓ 外部相似性文件已找到, 开始加载:")
                sno_external = pd.read_csv(sno_ext_path, index_col=0).values.astype(np.float32)
                disease_external = pd.read_csv(dis_ext_path, index_col=0).values.astype(np.float32)

                # 形状检查
                if sno_external.shape != self._snorna_gipk.shape:
                    raise ValueError(
                        f"snoRNA 外部相似性矩阵形状为 {sno_external.shape}, "
                        f"但 GIPK 相似性为 {self._snorna_gipk.shape}, 请检查行列顺序是否一致。"
                    )
                if disease_external.shape != self._disease_gipk.shape:
                    raise ValueError(
                        f"disease 外部相似性矩阵形状为 {disease_external.shape}, "
                        f"但 GIPK 相似性为 {self._disease_gipk.shape}, 请检查行列顺序是否一致。"
                    )

                self._sno_external = self._normalize_similarity(sno_external)
                self._disease_external = self._normalize_similarity(disease_external)
        else:
            print("  ✓ 使用缓存的关联矩阵与相似性（GIPK + 外部原始矩阵）")

        # ========== 每次调用都可以给不同 α ==========
        if alpha_snorna is None:
            alpha_snorna = self.alpha_snorna
        if alpha_disease is None:
            alpha_disease = self.alpha_disease

        # 没有外部相似性，就直接用 GIPK
        if (not self._external_available) or (self._sno_external is None) or (self._disease_external is None):
            snorna_sim_fused = self._snorna_gipk
            disease_sim_fused = self._disease_gipk
            print("\n  ✓ 使用 GIPK 相似性（未进行 SWF 融合）")
        else:
            snorna_sim_fused = self._swf_fusion(
                self._snorna_gipk, self._sno_external,
                name="snoRNA", alpha=alpha_snorna
            )
            disease_sim_fused = self._swf_fusion(
                self._disease_gipk, self._disease_external,
                name="Disease", alpha=alpha_disease
            )

            print("\n  ✓ SWF 融合完成:")
            print(f"    - snoRNA 相似性矩阵形状: {snorna_sim_fused.shape}")
            print(f"    - Disease 相似性矩阵形状: {disease_sim_fused.shape}")

        return (
            self._association_matrix.astype(np.float32),
            snorna_sim_fused.astype(np.float32),
            disease_sim_fused.astype(np.float32)
        )

    def _normalize_similarity(self, sim_matrix):
        """归一化相似性矩阵: 对称化 + 对角线设为 1 + 截断到 [0,1]"""
        sim_matrix = (sim_matrix + sim_matrix.T) / 2.0
        np.fill_diagonal(sim_matrix, 1.0)
        sim_matrix = np.clip(sim_matrix, 0, 1)
        return sim_matrix

    def _swf_fusion(self, sim_gipk, sim_ext, name="", alpha=None):
        """
        SWF 融合:
            如果 alpha 不为空:
                S_fused = alpha * sim_gipk + (1 - alpha) * sim_ext
            如果 alpha 为 None:
                使用“自动权重模式”（按均值算权重）
        """
        if sim_gipk.shape != sim_ext.shape:
            raise ValueError(
                f"SWF 融合失败: 两个矩阵形状不一致: {sim_gipk.shape} vs {sim_ext.shape}"
            )

        # 先做一次归一化和对称化
        sim_gipk = self._normalize_similarity(sim_gipk)
        sim_ext = self._normalize_similarity(sim_ext)

        if alpha is None:
            # 兼容旧逻辑: 按平均值自动算权重
            w_gipk = float(np.mean(sim_gipk))
            w_ext = float(np.mean(sim_ext))
            denom = w_gipk + w_ext + 1e-8
            alpha = w_gipk / denom
            print(f"    - [{name}] 自动计算 alpha(GIPK)={alpha:.4f} (基于均值)")
        else:
            alpha = float(alpha)
            alpha = max(0.0, min(1.0, alpha))

        print(
            f"    - [{name}] 最终使用 alpha(GIPK)={alpha:.4f}, "
            f"1-alpha(EXT)={1 - alpha:.4f}"
        )

        fused = alpha * sim_gipk + (1.0 - alpha) * sim_ext
        fused = self._normalize_similarity(fused)
        return fused


# 样本准备
def prepare_samples(association_matrix, train_indices, test_indices):  # 从关联矩阵中提取正负样本
    """从关联矩阵中提取正负样本"""
    """
    association_matrix：snoRNA-disease关联矩阵，形状 [num_snorna, num_disease]
    train_indices：训练集的正样本索引（来自K折交叉验证）
    test_indices：测试集的正样本索引（来自K折交叉验证）
    """
    num_snorna, num_disease = association_matrix.shape  # 获取矩阵维度

    # 找出所有正样本
    pos_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 1:
                pos_samples.append((i, j))

    pos_samples = np.array(pos_samples)

    # 划分正样本
    train_pos = pos_samples[train_indices]
    test_pos = pos_samples[test_indices]

    # 提取负样本
    neg_samples = []
    for i in range(num_snorna):
        for j in range(num_disease):
            if association_matrix[i, j] == 0:
                neg_samples.append((i, j))

    neg_samples = np.array(neg_samples)
    np.random.shuffle(neg_samples)  # 随机打乱负样本

    # 平衡采样负样本
    train_neg = neg_samples[:len(train_pos)]
    test_neg = neg_samples[len(train_pos):len(train_pos) + len(test_pos)]

    return train_pos, train_neg, test_pos, test_neg
