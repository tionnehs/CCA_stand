import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TrackFeatureCalculator:
    """单个track的特征计算器"""

    # 类常量
    CENTER_X = 960
    CENTER_Y = 540

    def __init__(self, track_df: pd.DataFrame, track_name: str = ""):
        self.track_df = track_df.copy()
        self.track_name = track_name
        self.features = {}

    def calculate_speed(self) -> 'TrackFeatureCalculator':
        """计算基于鼻子位置的速度"""
        self.track_df['nose_dx'] = self.track_df['nose1.x'].diff()
        self.track_df['nose_dy'] = self.track_df['nose1.y'].diff()
        frame_diff = self.track_df['frame_idx'].diff()
        self.track_df['speed'] = np.sqrt(
            self.track_df['nose_dx'] ** 2 + self.track_df['nose_dy'] ** 2
        ) / frame_diff.replace(0, np.nan)  # 避免除零
        return self

    def calculate_acceleration(self) -> 'TrackFeatureCalculator':
        """计算加速度"""
        if 'speed' not in self.track_df.columns:
            self.calculate_speed()
        frame_diff = self.track_df['frame_idx'].diff()
        self.track_df['acceleration'] = self.track_df['speed'].diff() / frame_diff.replace(0, np.nan)
        return self

    def calculate_body_length(self) -> 'TrackFeatureCalculator':
        """计算身体长度（鼻子到尾巴起点的距离）"""
        self.track_df['body_length'] = np.sqrt(
            (self.track_df['nose1.x'] - self.track_df['tailstart1.x']) ** 2 +
            (self.track_df['nose1.y'] - self.track_df['tailstart1.y']) ** 2
        )
        return self

    def calculate_deviation(self) -> 'TrackFeatureCalculator':
        """计算身体中心到场地中心的偏离程度"""
        centroid_x = (self.track_df['nose1.x'] + self.track_df['tailstart1.x']) / 2
        centroid_y = (self.track_df['nose1.y'] + self.track_df['tailstart1.y']) / 2
        self.track_df['deviation'] = np.sqrt(
            (centroid_x - self.CENTER_X) ** 2 + (centroid_y - self.CENTER_Y) ** 2
        )
        return self

    def calculate_head_width(self) -> 'TrackFeatureCalculator':
        """计算头部宽度（左右耳朵之间的距离）"""
        self.track_df['head_width'] = np.sqrt(
            (self.track_df['earL1.x'] - self.track_df['earR1.x']) ** 2 +
            (self.track_df['earL1.y'] - self.track_df['earR1.y']) ** 2
        )
        return self

    def calculate_movement_angle(self) -> 'TrackFeatureCalculator':
        """计算运动角度"""
        if 'nose_dx' not in self.track_df.columns or 'nose_dy' not in self.track_df.columns:
            self.calculate_speed()
        self.track_df['movement_angle'] = np.arctan2(
            self.track_df['nose_dy'], self.track_df['nose_dx']
        )
        return self

    def calculate_angular_velocity(self) -> 'TrackFeatureCalculator':
        """计算角速度"""
        if 'movement_angle' not in self.track_df.columns:
            self.calculate_movement_angle()
        self.track_df['angular_velocity'] = self.track_df['movement_angle'].diff()
        return self

    def calculate_quadrilateral_area(self) -> 'TrackFeatureCalculator':
        """计算四边形面积"""
        x_nose, y_nose = self.track_df['nose1.x'], self.track_df['nose1.y']
        x_earR, y_earR = self.track_df['earR1.x'], self.track_df['earR1.y']
        x_tail, y_tail = self.track_df['tailstart1.x'], self.track_df['tailstart1.y']
        x_earL, y_earL = self.track_df['earL1.x'], self.track_df['earL1.y']

        self.track_df['quad_area'] = 0.5 * np.abs(
            (x_nose * y_earR + x_earR * y_tail + x_tail * y_earL + x_earL * y_nose) -
            (y_nose * x_earR + y_earR * x_tail + y_tail * x_earL + y_earL * x_nose)
        )
        return self

    def calculate_quadrilateral_area_delta(self) -> 'TrackFeatureCalculator':
        """计算四边形面积变化率"""
        if 'quad_area' not in self.track_df.columns:
            self.calculate_quadrilateral_area()
        if 'speed' not in self.track_df.columns:
            self.calculate_speed()

        frame_diff = self.track_df['frame_idx'].diff()
        self.track_df['quad_area_delta'] = self.track_df['quad_area'].diff() / frame_diff.replace(0, np.nan)
        return self

    def calculate_head_tail_angle(self) -> 'TrackFeatureCalculator':
        """
        计算头部到鼻子和头部到尾巴的夹角（或曲率）

        """
        # 计算两个耳朵的中点（头部中心）
        head_center_x = (self.track_df['earL1.x'] + self.track_df['earR1.x']) / 2
        head_center_y = (self.track_df['earL1.y'] + self.track_df['earR1.y']) / 2

        # 计算头部中心到鼻子的向量
        head_nose_dx = self.track_df['nose1.x'] - head_center_x
        head_nose_dy = self.track_df['nose1.y'] - head_center_y

        # 计算头部中心到尾巴起点的向量
        head_tail_dx = self.track_df['tailstart1.x'] - head_center_x
        head_tail_dy = self.track_df['tailstart1.y'] - head_center_y

        # 计算向量点积
        dot_product = head_nose_dx * head_tail_dx + head_nose_dy * head_tail_dy

        # 计算向量模长
        mod_nose = np.sqrt(head_nose_dx ** 2 + head_nose_dy ** 2)
        mod_tail = np.sqrt(head_tail_dx ** 2 + head_tail_dy ** 2)

        # 避免除零错误
        mod_product = mod_nose * mod_tail
        mod_product = mod_product.replace(0, np.nan)

        # 计算夹角余弦值并转换为角度
        cos_angle = dot_product / mod_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保值在[-1, 1]范围内
        angle_degrees = np.arccos(cos_angle) * 180 / np.pi

        # 计算夹角本身
        self.track_df['head_angle'] = angle_degrees
        return self

    def calculate_all_features(self) -> 'TrackFeatureCalculator':
        """计算所有特征"""
        feature_methods = [
            self.calculate_speed,
            self.calculate_acceleration,
            self.calculate_body_length,
            self.calculate_deviation,
            self.calculate_head_width,
            self.calculate_movement_angle,
            self.calculate_angular_velocity,
            self.calculate_quadrilateral_area,
            self.calculate_quadrilateral_area_delta,
            self.calculate_head_tail_angle
        ]

        for method in feature_methods:
            method()

        if len(self.track_df) > 0:
            current_frame = self.track_df['frame_idx'].iloc[-1]
            print(f"[{self.track_name}] 所有特征计算完成，处理到第 {current_frame} 帧")
        else:
            print(f"[{self.track_name}] 所有特征计算完成，无有效帧数据")

        return self

    def get_summary_features(self) -> Dict:
        """获取特征汇总统计"""
        if len(self.track_df) == 0:
            return {'track_name': self.track_name, 'total_frames': 0}

        self.features = {
            'track_name': self.track_name,
            'total_frames': len(self.track_df),
            # 运动特征
            'avg_speed': self.track_df['speed'].mean(),
            'max_speed': self.track_df['speed'].max(),
            'speed_std': self.track_df['speed'].std(),
            'avg_acceleration': self.track_df['acceleration'].mean(),
            'max_acceleration': self.track_df['acceleration'].max(),
            # 身体形态特征
            'avg_body_length': self.track_df['body_length'].mean(),
            'body_length_std': self.track_df['body_length'].std(),
            'avg_head_width': self.track_df['head_width'].mean(),
            # 空间行为特征
            'avg_deviation': self.track_df['deviation'].mean(),
            'max_deviation': self.track_df['deviation'].max(),
            'avg_quad_area': self.track_df['quad_area'].mean(),
            'quad_area_std': self.track_df['quad_area'].std(),
            # 运动方向特征
            'avg_angular_velocity': self.track_df['angular_velocity'].mean(),
            'max_angular_velocity': self.track_df['angular_velocity'].max(),
            'avg_quad_area_delta': self.track_df['quad_area_delta'].mean(),
            'max_quad_area_delta': self.track_df['quad_area_delta'].max()
        }
        return self.features

    def save_to_csv(self, filename: Optional[str] = None) -> None:
        """保存处理后的数据到CSV"""
        if filename is None:
            filename = f"{self.track_name}_processed.csv"
        self.track_df.to_csv(filename, index=False)
        print(f"{self.track_name} 数据已保存到: {filename}")


class MultiTrackAnalyzer:
    """多track分析器"""

    def __init__(self):
        self.track_calculators = {}

    def load_data(self, csv_path: str) -> None:
        """从CSV文件加载数据"""
        df = pd.read_csv(csv_path)
        self.original_df = df

        # 分离两个track的数据
        track0_df = df[df['track'] == 'track_0'].copy()
        track1_df = df[df['track'] == 'track_1'].copy()

        self.track_calculators['track_0'] = TrackFeatureCalculator(track0_df, 'track_0')
        self.track_calculators['track_1'] = TrackFeatureCalculator(track1_df, 'track_1')

    def analyze_all_tracks(self) -> Dict[str, Dict]:
        """分析所有track"""
        results = {}
        for track_name, calculator in self.track_calculators.items():
            calculator.calculate_all_features()
            results[track_name] = calculator.get_summary_features()
        return results

    def get_track_data(self, track_name: str) -> pd.DataFrame:
        """获取指定track的处理后数据"""
        return self.track_calculators[track_name].track_df

    def save_all_tracks(self) -> None:
        """保存所有track的数据"""
        for calculator in self.track_calculators.values():
            calculator.save_to_csv()


def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建分析器并加载数据
    analyzer = MultiTrackAnalyzer()
    analyzer.load_data('Top_down_12mm_real.000_TrianOnly_2.analysis_washed.csv')

    # 分析所有track
    results = analyzer.analyze_all_tracks()

    # 打印结果
    for track_name, features in results.items():
        print(f"\n=== {track_name} 特征 ===")
        for key, value in features.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # 保存数据
    analyzer.save_all_tracks()
    print("\n数据处理完成并已保存!")


if __name__ == "__main__":
    main()