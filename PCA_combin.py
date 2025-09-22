import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class PCAAnalyzer:
    def __init__(self):
        # 自动PCA的跳过列（基础版本）
        self.auto_skip_columns = ['track', 'frame_idx', 'instance.score', 'nose1.x', 'nose1.y', 'nose1.score',
                                  'earL1.x', 'earL1.y', 'earL1.score', 'earR1.x', 'earR1.y', 'earR1.score',
                                  'tailstart1.x', 'tailstart1.y', 'tailstart1.score', 'tailend1.x', 'tailend1.y',
                                  'tailend1.score', 'nose_dx', 'nose_dy']

        # 手动筛选PCA的跳过列（扩展版本）
        self.manual_skip_columns = self.auto_skip_columns + ['acceleration', 'head_width', 'angular_velocity',
                                                             'body_length', 'quad_area_delta']

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def matrix_construct_auto(self, file_path):
        """
        自动PCA：使用基础的特征选择构建特征矩阵
        """
        return self.matrix_construct(file_path, self.auto_skip_columns)

    def matrix_construct_manual(self, file_path):
        """
        手动筛选PCA：使用扩展的特征选择构建特征矩阵
        """
        return self.matrix_construct(file_path, self.manual_skip_columns)

    def matrix_construct(self, file_path, skip_columns):
        """
        通用的矩阵构建函数
        """
        df = pd.read_csv(file_path)

        # 提取需要用于PCA的特征列（跳过指定列）
        feature_columns = [col for col in df.columns if col not in skip_columns]

        # 检查并删除包含空值的行
        initial_rows = len(df)
        df_clean = df.dropna(subset=feature_columns)
        removed_rows = initial_rows - len(df_clean)

        if removed_rows > 0:
            print(f"从 {file_path} 中删除了 {removed_rows} 行包含空值的数据")

        # 如果没有有效数据，返回None
        if len(df_clean) == 0:
            print(f"警告: {file_path} 中没有有效数据")
            return None

        # 提取跳过指定列后的所有数据作为PCA输入矩阵
        feature_nor_matrix = df_clean[feature_columns].values

        return feature_nor_matrix

    def perform_pca(self, matrix, n_components=2, normalize=True):
        """
        对输入矩阵进行PCA降维
        """
        if matrix is None:
            print("警告: 输入矩阵为空")
            return None

        print(f"\n=== 开始{n_components}维降维 ===")

        # 可选：对数据进行标准化
        if normalize:
            scaler = StandardScaler()
            matrix_normalized = scaler.fit_transform(matrix)
            print("数据已进行标准化处理")
        else:
            matrix_normalized = matrix
            print("使用原始数据进行PCA")

        # 创建PCA对象并进行降维
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(matrix_normalized)

        # 获取PCA的各种信息
        result = {
            'reduced_data': reduced_data,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'components': pca.components_,
            'mean': pca.mean_,
            'n_components': pca.n_components_,
            'total_variance_ratio': sum(pca.explained_variance_ratio_),
            'normalized': normalize
        }

        return result

    def prepare_comparison_data(self, file_paths, group_names, pca_type='auto'):
        """
        准备用于比较的合并数据
        pca_type: 'auto' 或 'manual'
        """
        all_features = []
        all_labels = []
        feature_names = None

        # 选择对应的矩阵构建函数
        if pca_type == 'auto':
            matrix_func = self.matrix_construct_auto
            skip_columns = self.auto_skip_columns
        elif pca_type == 'manual':
            matrix_func = self.matrix_construct_manual
            skip_columns = self.manual_skip_columns
        else:
            raise ValueError("pca_type 必须是 'auto' 或 'manual'")

        for i, file_path in enumerate(file_paths):
            matrix = matrix_func(file_path)
            if matrix is None:
                continue

            if feature_names is None:
                # 获取特征名称
                df_temp = pd.read_csv(file_path)
                feature_names = [col for col in df_temp.columns if col not in skip_columns]

            # 添加数据
            all_features.append(matrix)
            # 添加标签
            all_labels.extend([group_names[i]] * matrix.shape[0])

        # 合并所有数据
        if not all_features:
            print("警告: 没有有效数据")
            return None, None, None

        combined_matrix = np.vstack(all_features)
        labels = np.array(all_labels)

        return combined_matrix, labels, feature_names

    def analyze_common_specific_features(self, pca_result, feature_names, group1_name, group2_name, top_n=5):
        """
        分析共性特征和特异性特征
        """
        components = pca_result['components']
        n_components = pca_result['n_components']

        print("\n=== 特征重要性分析 ===")

        for i in range(n_components):
            print(f"\n主成分 {i + 1} 的特征重要性:")

            # 获取该主成分的载荷
            loadings = components[i]

            # 按绝对值排序
            feature_importance = sorted(zip(feature_names, loadings),
                                        key=lambda x: abs(x[1]), reverse=True)

            # 打印最重要的特征
            print(f"前{top_n}个最重要特征:")
            for feature, loading in feature_importance[:top_n]:
                print(f"  {feature}: {loading:.4f}")

            # 分析正负载荷的特征
            positive_features = [(f, l) for f, l in feature_importance if l > 0][:3]
            negative_features = [(f, l) for f, l in feature_importance if l < 0][:3]

            print(f"正载荷特征（可能与{group1_name}相关）:")
            for feature, loading in positive_features:
                print(f"  {feature}: {loading:.4f}")

            print(f"负载荷特征（可能与{group2_name}相关）:")
            for feature, loading in negative_features:
                print(f"  {feature}: {loading:.4f}")

    def plot_comparison_results(self, pca_result, labels, group_names, save_path=None):
        """
        绘制PCA比较结果并保存
        """
        reduced_data = pca_result['reduced_data']

        plt.figure(figsize=(12, 5))

        # 绘制散点图
        plt.subplot(1, 2, 1)
        colors = ['red', 'blue']
        for i, group in enumerate(group_names):
            mask = labels == group
            plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                        alpha=0.6, label=group, c=colors[i])

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Comparison')
        plt.legend()
        plt.grid(True)

        # 绘制方差贡献率
        plt.subplot(1, 2, 2)
        explained_variance = pca_result['explained_variance_ratio']
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Ratio')
        plt.title('Explained Variance Ratio')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
        plt.close()

    def plot_variance_analysis(self, pca_result, save_path=None):
        """
        绘制方差分析图：前n个主成分的累积方差贡献率并保存
        """
        explained_variance_ratio = pca_result['explained_variance_ratio']
        n_components = len(explained_variance_ratio)

        # 计算累积方差贡献率
        cumulative_variance = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(12, 5))

        # 子图1：各个主成分的方差贡献率
        plt.subplot(1, 2, 1)
        plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.6, color='skyblue')
        plt.xlabel('主成分序号')
        plt.ylabel('方差贡献率')
        plt.title('各个主成分的方差贡献率')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(explained_variance_ratio):
            plt.text(i + 1, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # 子图2：累积方差贡献率
        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_components + 1), cumulative_variance, 'o-', linewidth=2, markersize=8, color='orange')
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%方差线')
        plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%方差线')
        plt.axhline(y=0.95, color='b', linestyle='--', alpha=0.7, label='95%方差线')

        plt.xlabel('前n个主成分')
        plt.ylabel('累积方差贡献率')
        plt.title('前n个主成分的累积方差贡献率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        # 添加数值标签
        for i, v in enumerate(cumulative_variance):
            plt.text(i + 1, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
        plt.close()

        # 打印详细信息
        print("=== 方差分析详细信息 ===")
        for i in range(n_components):
            print(f"前{i + 1}个主成分累积方差: {cumulative_variance[i]:.4f} ({cumulative_variance[i] * 100:.2f}%)")

        # 找到达到不同阈值所需的主成分数量
        thresholds = [0.8, 0.9, 0.95, 0.99]
        for threshold in thresholds:
            n_needed = np.argmax(cumulative_variance >= threshold) + 1
            if cumulative_variance[n_needed - 1] >= threshold:
                print(f"达到{threshold * 100:.0f}%方差需要前{n_needed}个主成分")
            else:
                print(f"无法达到{threshold * 100:.0f}%方差，最高为{cumulative_variance[-1] * 100:.2f}%")

    def plot_scree_plot(self, pca_result, save_path=None):
        """
        绘制碎石图（Scree Plot）：显示各个特征值的大小并保存
        """
        explained_variance = pca_result['explained_variance']
        n_components = len(explained_variance)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_components + 1), explained_variance, 'o-', linewidth=2, markersize=8)
        plt.xlabel('主成分序号')
        plt.ylabel('特征值（方差）')
        plt.title('碎石图 (Scree Plot)')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(explained_variance):
            plt.text(i + 1, v + max(explained_variance) * 0.02, f'{v:.2f}', ha='center', va='bottom')

        # 添加特征值=1的参考线（Kaiser准则）
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='特征值=1（Kaiser准则）')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")
        plt.close()

    def main_comparison_auto(self, output_dir="pca_auto_results"):
        """
        自动PCA比较分析
        """
        return self.main_comparison(output_dir, 'auto')

    def main_comparison_manual(self, output_dir="pca_manual_results"):
        """
        手动筛选PCA比较分析
        """
        return self.main_comparison(output_dir, 'manual')

    def main_comparison(self, output_dir, pca_type):
        """
        通用的比较分析函数
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 设置文件路径和组名
        file_paths = ["track_0_processed_normalized.csv", "track_1_processed_normalized.csv"]
        group_names = ["Group1", "Group2"]

        print(f"开始{pca_type} PCA比较分析...")

        # 准备合并数据
        combined_matrix, labels, feature_names = self.prepare_comparison_data(file_paths, group_names, pca_type)

        if combined_matrix is None:
            print("无法准备比较数据")
            return None

        print(f"合并后的数据形状: {combined_matrix.shape}")
        print(f"组别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"使用的特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")

        # 进行PCA分析 - 使用所有主成分来查看完整的方差分布
        pca_result = self.perform_pca(combined_matrix, n_components=min(20, combined_matrix.shape[1]), normalize=True)

        if pca_result is None:
            print("PCA分析失败")
            return None

        # 绘制方差分析图并保存
        variance_plot_path = os.path.join(output_dir, "variance_analysis.png")
        self.plot_variance_analysis(pca_result, save_path=variance_plot_path)

        # 绘制碎石图并保存
        scree_plot_path = os.path.join(output_dir, "scree_plot.png")
        self.plot_scree_plot(pca_result, save_path=scree_plot_path)

        # 分析共性特征和特异性特征（只分析前5个主成分）
        pca_5d_result = self.perform_pca(combined_matrix, n_components=5, normalize=True)
        self.analyze_common_specific_features(pca_5d_result, feature_names, group_names[0], group_names[1])

        # 绘制比较结果并保存
        comparison_plot_path = os.path.join(output_dir, "pca_comparison.png")
        self.plot_comparison_results(pca_5d_result, labels, group_names, save_path=comparison_plot_path)

        return {
            'pca_full_result': pca_result,  # 包含所有主成分的完整结果
            'pca_5d_result': pca_5d_result,  # 前5个主成分的结果
            'labels': labels,
            'feature_names': feature_names,
            'combined_matrix': combined_matrix,
            'output_dir': output_dir,
            'pca_type': pca_type
        }

    def plot_dimension_comparison(self, auto_matrix, manual_matrix, save_path=None):
        """
        绘制自动检测vs手动标记的维度对比图（达到90%方差所需的主成分数量）

        参数:
        auto_matrix: 自动检测特征矩阵
        manual_matrix: 手动标记特征矩阵
        save_path: 图像保存路径
        """
        # 计算自动检测数据达到90%方差所需的主成分数
        pca_auto = PCA()
        pca_auto.fit(StandardScaler().fit_transform(auto_matrix))
        cumulative_variance_auto = np.cumsum(pca_auto.explained_variance_ratio_)
        n_components_auto = np.argmax(cumulative_variance_auto >= 0.9) + 1

        # 计算手动标记数据达到90%方差所需的主成分数
        pca_manual = PCA()
        pca_manual.fit(StandardScaler().fit_transform(manual_matrix))
        cumulative_variance_manual = np.cumsum(pca_manual.explained_variance_ratio_)
        n_components_manual = np.argmax(cumulative_variance_manual >= 0.9) + 1

        # 创建图表
        plt.figure(figsize=(8, 6))

        # 设置柱状图位置和宽度
        x_pos = [0, 1]
        values = [n_components_auto, n_components_manual]
        labels = ['Auto-tracking', 'Manual\nAnnotation']
        colors = ['#2E86AB', '#A23B72']  # 蓝色和紫红色

        # 绘制柱状图
        bars = plt.bar(x_pos, values, color=colors, alpha=0.8, width=0.6)

        # 在每个柱子上添加数值标签
        for i, v in enumerate(values):
            plt.text(x_pos[i], v + 0.1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 设置图表属性
        plt.ylabel('Number of dimensions\n(90% variance explained)', fontsize=12)
        plt.xticks(x_pos, labels, fontsize=11)
        plt.title('Dimensionality Comparison: Auto-tracking vs Manual Annotation', fontsize=14, pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # 设置y轴范围，留出一些空间给数值标签
        plt.ylim(0, max(values) * 1.2)

        # 添加一些统计信息到图例
        auto_variance = cumulative_variance_auto[n_components_auto - 1]
        manual_variance = cumulative_variance_manual[n_components_manual - 1]

        info_text = f'Auto: {n_components_auto} dims → {auto_variance:.1%} variance\nManual: {n_components_manual} dims → {manual_variance:.1%} variance'
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"维度对比图已保存至: {save_path}")

        plt.show()

        return {
            'auto_components': n_components_auto,
            'manual_components': n_components_manual,
            'auto_variance': auto_variance,
            'manual_variance': manual_variance
        }


# 程序入口
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = PCAAnalyzer()

    # 执行自动PCA分析
    print("=" * 50)
    print("执行自动PCA分析")
    print("=" * 50)
    auto_results = analyzer.main_comparison_auto("pca_auto_results")

    # 执行手动筛选PCA分析
    print("\n" + "=" * 50)
    print("执行手动筛选PCA分析")
    print("=" * 50)
    manual_results = analyzer.main_comparison_manual("pca_manual_results")

    # 显示结果摘要
    if auto_results is not None:
        print(f"\n自动PCA分析完成，结果保存在: {auto_results['output_dir']}")

    if manual_results is not None:
        print(f"手动PCA分析完成，结果保存在: {manual_results['output_dir']}")

    print("\n=== 所有分析完成 ===")