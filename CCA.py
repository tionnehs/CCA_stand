import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os


class CCAAnalyzer:
    def __init__(self, output_dir="cca_results"):
        """
        初始化CCA分析器

        Parameters:
        -----------
        output_dir : str, optional
            输出目录路径，默认为"cca_results"
        """
        self.output_dir = output_dir
        self.skip_columns = [
            'track', 'frame_idx', 'instance.score', 'nose1.x', 'nose1.y', 'nose1.score',
            'earL1.x', 'earL1.y', 'earL1.score', 'earR1.x', 'earR1.y', 'earR1.score',
            'tailstart1.x', 'tailstart1.y', 'tailstart1.score', 'tailend1.x', 'tailend1.y',
            'tailend1.score', 'nose_dx', 'nose_dy', 'quad_centroid_y', 'quad_centroid_x'
        ]

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, file_path):
        """
        从CSV文件加载数据并构建特征矩阵

        Parameters:
        -----------
        file_path : str
            CSV文件路径

        Returns:
        --------
        matrix : numpy.ndarray or None
            特征矩阵
        feature_names : list or None
            特征名称列表
        """
        try:
            df = pd.read_csv(file_path)
            feature_columns = [col for col in df.columns if col not in self.skip_columns]

            # 删除包含空值的行
            df_clean = df.dropna(subset=feature_columns)

            if len(df_clean) == 0:
                print(f"警告: {file_path} 中没有有效数据")
                return None, None

            return df_clean[feature_columns].values, feature_columns

        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None, None

    def perform_cca(self, matrix1, matrix2, n_components=6):
        """
        执行典型相关分析(CCA)

        Parameters:
        -----------
        matrix1 : numpy.ndarray
            第一组特征矩阵
        matrix2 : numpy.ndarray
            第二组特征矩阵
        n_components : int, optional
            要提取的典型变量数量，默认为6

        Returns:
        --------
        result : dict
            CCA分析结果字典
        """
        # 数据标准化
        scaler = StandardScaler()
        m1_scaled = scaler.fit_transform(matrix1)
        m2_scaled = scaler.fit_transform(matrix2)

        # 执行CCA
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(m1_scaled, m2_scaled)

        # 计算共性特征强度：两组载荷平均绝对值
        feature_common_X = np.mean(np.abs(cca.x_weights_), axis=1)
        feature_common_Y = np.mean(np.abs(cca.y_weights_), axis=1)
        feature_common = (feature_common_X + feature_common_Y) / 2

        # 计算特异性特征强度：残差绝对值和
        feature_specific_X = np.sum(np.abs(m1_scaled - X_c @ cca.x_loadings_.T), axis=0)
        feature_specific_Y = np.sum(np.abs(m2_scaled - Y_c @ cca.y_loadings_.T), axis=0)

        result = {
            "cca_model": cca,
            "X_transformed": X_c,
            "Y_transformed": Y_c,
            "feature_common_scores": feature_common,
            "feature_specific_scores_X": feature_specific_X,
            "feature_specific_scores_Y": feature_specific_Y,
            "x_weights": cca.x_weights_,
            "y_weights": cca.y_weights_,
            "n_components": n_components
        }

        return result

    def extract_top_features(self, feature_scores, top_k=5, feature_names=None):
        """
        提取最重要的特征

        Parameters:
        -----------
        feature_scores : numpy.ndarray
            特征重要性分数
        top_k : int, optional
            要提取的顶部特征数量，默认为5
        feature_names : list, optional
            特征名称列表

        Returns:
        --------
        result : dict
            包含顶部特征信息的结果字典
        """
        top_indices = np.argsort(-feature_scores)[:top_k]
        top_scores = feature_scores[top_indices]

        result = {
            "indices": top_indices,
            "scores": top_scores
        }

        if feature_names is not None:
            result["names"] = [feature_names[i] for i in top_indices]
            result["feature_score_pairs"] = list(zip([feature_names[i] for i in top_indices], top_scores))

        return result

    def plot_feature_timeseries(self, matrix1, matrix2, feature_indices, feature_names,
                                group_names=("Group1", "Group2"), feature_type="Common"):
        """
        绘制特征时间序列图（单个 + 合并大图）

        Parameters:
        -----------
        matrix1 : numpy.ndarray
            第一组数据矩阵
        matrix2 : numpy.ndarray
            第二组数据矩阵
        feature_indices : list
            要绘制的特征索引列表
        feature_names : list
            特征名称列表
        group_names : tuple, optional
            组名元组，默认为("Group1", "Group2")
        feature_type : str, optional
            特征类型（用于文件名），默认为"Common"
        """
        # ===== 保留单独绘图 =====
        scaler = MinMaxScaler()
        for idx in feature_indices:
            feature_name = feature_names[idx]

            # 用同一个 scaler 对两组进行归一化，保持可比性
            all_series = np.concatenate([matrix1[:, idx], matrix2[:, idx]]).reshape(-1, 1)
            scaler.fit(all_series)
            series1_normalized = scaler.transform(matrix1[:, idx].reshape(-1, 1)).flatten()
            series2_normalized = scaler.transform(matrix2[:, idx].reshape(-1, 1)).flatten()

            # 单张图
            plt.figure(figsize=(10, 5))
            plt.plot(series1_normalized, label=group_names[0], linestyle='-', alpha=0.8)
            plt.plot(series2_normalized, label=group_names[1], linestyle='-', alpha=0.8)

            plt.xlabel("帧索引", fontsize=12)
            plt.ylabel("归一化强度", fontsize=12)
            plt.title(f"{feature_type}特征: {feature_name}", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            # 保存单个图
            filename = f"{feature_type}_{feature_name.replace(' ', '_')}.png"
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"已保存 {feature_type} 特征图: {save_path}")

        # ===== 生成合并大图 =====
        fig, axes = plt.subplots(nrows=len(feature_indices), ncols=1,
                                 figsize=(10, 4 * len(feature_indices)),
                                 sharex=True)

        if len(feature_indices) == 1:  # 只有一个特征时，axes 不是数组
            axes = [axes]

        for ax, idx in zip(axes, feature_indices):
            feature_name = feature_names[idx]

            all_series = np.concatenate([matrix1[:, idx], matrix2[:, idx]]).reshape(-1, 1)
            scaler.fit(all_series)
            series1_normalized = scaler.transform(matrix1[:, idx].reshape(-1, 1)).flatten()
            series2_normalized = scaler.transform(matrix2[:, idx].reshape(-1, 1)).flatten()

            ax.plot(series1_normalized, label=group_names[0], linestyle='-', alpha=0.8)
            ax.plot(series2_normalized, label=group_names[1], linestyle='-', alpha=0.8)
            ax.set_title(f"{feature_type}特征: {feature_name}", fontsize=12)
            ax.set_ylabel("归一化强度")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("帧索引")
        axes[0].legend(fontsize=10)

        plt.tight_layout()
        big_filename = f"{feature_type}_AllFeatures.png"
        big_save_path = os.path.join(self.output_dir, big_filename)
        plt.savefig(big_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存 {feature_type} 合并特征图: {big_save_path}")

    def plot_correlation_strength(self, common_features, specific_features, feature_names):
        """
        绘制特征相关性强度图

        Parameters:
        -----------
        common_features : dict
            共性特征信息
        specific_features : dict
            特异性特征信息
        feature_names : list
            特征名称列表
        """
        plt.figure(figsize=(12, 8))

        # 创建所有特征的分数数组
        all_scores = np.zeros(len(feature_names))
        all_scores[common_features["indices"]] = common_features["scores"]

        # 选择要显示的前20个特征
        display_indices = np.argsort(-all_scores)[:20]
        display_scores = all_scores[display_indices]
        display_names = [feature_names[i] for i in display_indices]

        # 创建颜色映射：共性特征为蓝色，其他为灰色
        colors = ['blue' if i in common_features["indices"] else 'gray' for i in display_indices]

        plt.barh(range(len(display_names)), display_scores, color=colors)
        plt.yticks(range(len(display_names)), display_names)
        plt.xlabel('特征重要性分数')
        plt.title('特征相关性强度图（蓝色为共性特征）')
        plt.gca().invert_yaxis()  # 反转y轴使最高分数在顶部

        save_path = os.path.join(self.output_dir, "feature_correlation_strength.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存特征相关性强度图: {save_path}")

    def analyze(self, file_paths, group_names=None, top_k=5):
        """
        执行完整的CCA分析流程

        Parameters:
        -----------
        file_paths : list
            包含两个文件路径的列表
        group_names : list, optional
            组名列表，默认为["Group1", "Group2"]
        top_k : int, optional
            要提取的顶部特征数量，默认为5

        Returns:
        --------
        results : dict
            分析结果字典
        """
        if group_names is None:
            group_names = ["Group1", "Group2"]

        print("=" * 60)
        print("开始CCA分析")
        print("=" * 60)

        # 加载数据
        print("加载数据...")
        matrix1, feature_names1 = self.load_data(file_paths[0])
        matrix2, feature_names2 = self.load_data(file_paths[1])

        if matrix1 is None or matrix2 is None:
            print("数据加载失败，请检查文件路径和数据格式")
            return None

        print(f"数据1形状: {matrix1.shape}")
        print(f"数据2形状: {matrix2.shape}")
        print(f"特征数量: {len(feature_names1)}")

        # 执行CCA
        print("执行典型相关分析...")
        cca_result = self.perform_cca(matrix1, matrix2, n_components=top_k)

        # 提取共性特征
        print("提取共性特征...")
        common_features = self.extract_top_features(
            cca_result["feature_common_scores"],
            top_k=top_k,
            feature_names=feature_names1
        )

        # 提取特异性特征
        print("提取特异性特征...")
        specific_scores = cca_result["feature_specific_scores_X"] + cca_result["feature_specific_scores_Y"]
        specific_features = self.extract_top_features(
            specific_scores,
            top_k=top_k,
            feature_names=feature_names1
        )

        # 打印结果
        print("\n" + "=" * 60)
        print("分析结果")
        print("=" * 60)

        print("\n=== 共性特征 (前5名) ===")
        for i, (name, score) in enumerate(common_features["feature_score_pairs"], 1):
            print(f"{i}. {name}: {score:.4f}")

        print("\n=== 特异性特征 (前5名) ===")
        for i, (name, score) in enumerate(specific_features["feature_score_pairs"], 1):
            print(f"{i}. {name}: {score:.4f}")

        # 绘制特征时间序列图
        print("\n生成可视化图表...")
        self.plot_feature_timeseries(
            matrix1, matrix2,
            common_features["indices"],
            feature_names1,
            group_names=group_names,
            feature_type="Common"
        )

        self.plot_feature_timeseries(
            matrix1, matrix2,
            specific_features["indices"],
            feature_names1,
            group_names=group_names,
            feature_type="Specific"
        )

        # 绘制相关性强度图
        self.plot_correlation_strength(common_features, specific_features, feature_names1)

        # 返回完整结果
        results = {
            "cca_result": cca_result,
            "common_features": common_features,
            "specific_features": specific_features,
            "feature_names": feature_names1,
            "matrix1": matrix1,
            "matrix2": matrix2,
            "output_dir": self.output_dir,
            "group_names": group_names
        }

        print(f"\n分析完成！结果已保存到: {self.output_dir}")
        return results


# 使用示例
if __name__ == "__main__":
    # 创建CCA分析器实例
    cca_analyzer = CCAAnalyzer(output_dir="cca_analysis_results")

    # 执行分析
    file_paths = ["track_0_processed_normalized.csv", "track_1_processed_normalized.csv"]
    group_names = ["Track0", "Track1"]

    results = cca_analyzer.analyze(
        file_paths=file_paths,
        group_names=group_names,
        top_k=5
    )

    if results is not None:
        print("\n=== 分析摘要 ===")
        print(f"共性特征数量: {len(results['common_features']['indices'])}")
        print(f"特异性特征数量: {len(results['specific_features']['indices'])}")
        print(f"输出目录: {results['output_dir']}")