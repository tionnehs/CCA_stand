import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import glob
import warnings

# 忽略特定的警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class MouseTrackNormalizer:
    def __init__(self):
        self.scalers = {}
        self.track_files = []
        self.normalized_dfs = {}

    def find_track_files(self, pattern='track_*_processed.csv'):
        """查找所有符合模式的轨迹文件"""
        self.track_files = glob.glob(pattern)
        return self.track_files

    def load_and_normalize(self, file_path):
        """加载并归一化单个轨迹文件"""
        # 定义需要跳过的列（非特征列）
        skip_columns = ['track', 'frame_idx', 'instance.score', 'nose1.x', 'nose1.y', 'nose1.score',
                        'earL1.x', 'earL1.y', 'earL1.score', 'earR1.x', 'earR1.y', 'earR1.score',
                        'tailstart1.x', 'tailstart1.y', 'tailstart1.score', 'tailend1.x', 'tailend1.y',
                        'tailend1.score']

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取需要归一化的特征列
        feature_columns = [col for col in df.columns if col not in skip_columns]

        # 检查并删除包含空值的行
        initial_rows = len(df)
        df_clean = df.dropna(subset=feature_columns).copy()  # 添加.copy()避免链式赋值警告
        removed_rows = initial_rows - len(df_clean)

        if removed_rows > 0:
            print(f"从 {file_path} 中删除了 {removed_rows} 行包含空值的数据")

        # 如果没有有效数据，返回None
        if len(df_clean) == 0:
            print(f"警告: {file_path} 中没有有效数据")
            return None

        # 初始化MinMaxScaler并归一化特征列
        scaler = MinMaxScaler()
        # 使用.loc避免SettingWithCopyWarning
        normalized_features = scaler.fit_transform(df_clean[feature_columns])
        df_clean.loc[:, feature_columns] = normalized_features

        # 保存scaler供后续使用
        track_name = os.path.basename(file_path).replace('_processed.csv', '')
        self.scalers[track_name] = scaler

        return df_clean

    def normalize_all_tracks(self):
        """归一化所有找到的轨迹文件"""
        self.normalized_dfs = {}

        for file_path in self.track_files:
            track_name = os.path.basename(file_path).replace('_processed.csv', '')
            print(f"正在处理: {track_name}")

            normalized_df = self.load_and_normalize(file_path)
            if normalized_df is not None:
                self.normalized_dfs[track_name] = normalized_df

                # 保存归一化后的文件
                output_file = file_path.replace('.csv', '_normalized.csv')
                normalized_df.to_csv(output_file, index=False)
                print(f"{track_name} 归一化完成，已保存为 {output_file}")
            else:
                print(f"{track_name} 处理失败")

        return self.normalized_dfs

    def get_normalized_data(self):
        """获取归一化后的数据"""
        return self.normalized_dfs

    def get_scalers(self):
        """获取标准化器"""
        return self.scalers


class MouseTrackVisualizer:
    def __init__(self, font_settings=None):
        """初始化可视化器"""
        if font_settings is None:
            font_settings = {
                'sans-serif': ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei'],
                'axes.unicode_minus': False
            }

        # 设置字体
        plt.rcParams['font.sans-serif'] = font_settings['sans-serif']
        plt.rcParams['axes.unicode_minus'] = font_settings['axes.unicode_minus']

    def plot_normalized_features(self, normalized_dfs, feature_name='speed', save_path=None):
        """绘制归一化后的特征折线图"""
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        has_valid_data = False

        for i, (track_name, df) in enumerate(normalized_dfs.items()):
            if feature_name in df.columns:
                has_valid_data = True
                color = colors[i % len(colors)]
                plt.plot(df['frame_idx'], df[feature_name],
                         label=track_name, color=color, linewidth=1.5)

                # 添加数据点标记
                if len(df) < 100:  # 只在数据点较少时添加标记
                    plt.scatter(df['frame_idx'], df[feature_name],
                                color=color, s=20, alpha=0.7)

        if not has_valid_data:
            print(f"警告: 没有找到特征 '{feature_name}' 的数据")
            plt.close()
            return

        plt.xlabel('帧数')
        plt.ylabel(feature_name)
        plt.title('归一化特征随时间变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存图表或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_multiple_features(self, normalized_dfs, feature_names, save_path=None):
        """绘制多个特征的子图"""
        n_features = len(feature_names)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))

        if n_features == 1:
            axes = [axes]

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for j, feature_name in enumerate(feature_names):
            ax = axes[j]
            has_valid_data = False

            for i, (track_name, df) in enumerate(normalized_dfs.items()):
                if feature_name in df.columns:
                    has_valid_data = True
                    color = colors[i % len(colors)]
                    ax.plot(df['frame_idx'], df[feature_name],
                            label=track_name, color=color, linewidth=1.5)

            if has_valid_data:
                ax.set_xlabel('frame')
                ax.set_ylabel(feature_name)
                ax.set_title(f'{feature_name} ')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f'{feature_name} - 无数据')
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # 保存图表或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_distribution(self, normalized_dfs, feature_name='speed', save_path=None):
        """绘制特征分布直方图"""
        plt.figure(figsize=(10, 6))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        has_valid_data = False

        for i, (track_name, df) in enumerate(normalized_dfs.items()):
            if feature_name in df.columns:
                has_valid_data = True
                color = colors[i % len(colors)]
                plt.hist(df[feature_name], bins=30, alpha=0.5,
                         label=track_name, color=color)

        if not has_valid_data:
            print(f"警告: 没有找到特征 '{feature_name}' 的数据")
            plt.close()
            return

        plt.xlabel(feature_name)
        plt.ylabel('频率')
        plt.title(f'{feature_name} 分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 创建归一化器实例并处理数据
    normalizer = MouseTrackNormalizer()

    # 查找轨迹文件
    track_files = normalizer.find_track_files('track*_processed.csv')

    if not track_files:
        print("未找到任何轨迹文件")
        return

    print(f"找到 {len(track_files)} 个轨迹文件: {track_files}")

    # 归一化所有轨迹
    normalized_dfs = normalizer.normalize_all_tracks()

    if not normalized_dfs:
        print("没有成功归一化的轨迹文件")
        return

    # 创建可视化器实例并生成图表
    visualizer = MouseTrackVisualizer()

    print("\n生成可视化图表...")

    # 检查第一个数据框的列，确保特征存在
    first_df = list(normalized_dfs.values())[0]
    available_features = [col for col in first_df.columns if col not in ['track', 'frame_idx']]
    print(f"可用的特征: {available_features}")

    # 绘制单个特征的折线图（使用实际存在的特征）
    if 'speed' in available_features:
        visualizer.plot_normalized_features(
            normalized_dfs,
            feature_name='speed',
            save_path='normalized_speed_comparison.png'
        )
    else:
        print("警告: 'speed' 特征不存在，跳过绘制")

    # 绘制多个特征的子图（只使用实际存在的特征）
    possible_feature_names = ['speed', 'acceleration', 'body_length', 'deviation',
                              'head_width', 'movement_angle', 'angular_velocity',
                              'quad_centroid_x', 'quad_centroid_y', 'quad_area',
                              'quadrilateral_area_delta', 'head_tail_angle']

    # 筛选出实际存在的特征
    existing_features = [feat for feat in possible_feature_names if feat in available_features]

    if existing_features:
        visualizer.plot_multiple_features(
            normalized_dfs,
            feature_names=existing_features[:4],  # 限制特征数量，避免图表过多
            save_path='multiple_features_comparison.png'
        )
    else:
        print("警告: 没有找到任何指定的特征，跳过多特征图表")

    # 绘制特征分布图
    if 'speed' in available_features:
        visualizer.plot_feature_distribution(
            normalized_dfs,
            feature_name='speed',
            save_path='speed_distribution.png'
        )
    else:
        print("警告: 'speed' 特征不存在，跳过分布图")

    print("处理完成！所有图表已保存为图像文件。")


if __name__ == "__main__":
    main()