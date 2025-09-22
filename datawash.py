import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional


class TrackDataValidator:
    """验证单条track数据完整性的类"""

    REQUIRED_COLS = ['nose1', 'earL1', 'earR1', 'tailstart1']
    OPTIONAL_COLS = ['tailend1']

    @classmethod
    def is_track_valid(cls, track_row: pd.Series) -> bool:
        """检查单条track数据是否完整"""
        for part in cls.REQUIRED_COLS:
            if (pd.isna(track_row[f'{part}.score']) or
                    pd.isna(track_row[f'{part}.x']) or
                    pd.isna(track_row[f'{part}.y'])):
                return False
        return True


class FrameValidator:
    """验证单帧数据完整性的类"""

    @staticmethod
    def is_frame_valid(frame_data: pd.DataFrame) -> bool:
        """检查该帧数据是否有效"""
        # 检查该帧是否包含两个track
        tracks_in_frame = frame_data['track'].unique()
        if len(tracks_in_frame) != 2 or not all(track in tracks_in_frame for track in ['track_0', 'track_1']):
            return False

        # 分别检查track0和track1的数据完整性
        for track in ['track_0', 'track_1']:
            track_data = frame_data[frame_data['track'] == track]
            if len(track_data) != 1:  # 每个track应该只有一行数据
                return False

            if not TrackDataValidator.is_track_valid(track_data.iloc[0]):
                return False

        return True


class ConcurrentTrackFilter:
    """并发track数据过滤器"""

    def __init__(self, csv_file_path: str, output_suffix: str = "_washed"):
        self.csv_file_path = csv_file_path
        self.output_suffix = output_suffix
        self.df = None
        self.filtered_df = None
        self.output_file_path = self._generate_output_path()

    def _generate_output_path(self) -> str:
        """生成输出文件路径"""
        file_dir = os.path.dirname(self.csv_file_path)
        file_name = os.path.basename(self.csv_file_path)
        name_without_ext, ext = os.path.splitext(file_name)    # 分割文件名
        return os.path.join(file_dir, f"{name_without_ext}{self.output_suffix}{ext}")

    def load_data(self) -> None:
        """加载CSV数据"""
        self.df = pd.read_csv(self.csv_file_path)

    def _get_frames_with_both_tracks(self) -> List[int]:
        """获取同时存在两个track的帧"""
        frame_counts = self.df.groupby('frame_idx')['track'].nunique()
        return frame_counts[frame_counts == 2].index.tolist()

    def _filter_concurrent_frames(self) -> pd.DataFrame:
        """过滤出同时有track0和track1的帧"""
        frames_with_both_tracks = self._get_frames_with_both_tracks()
        return self.df[self.df['frame_idx'].isin(frames_with_both_tracks)].copy()

    def _get_valid_frames(self, concurrent_df: pd.DataFrame) -> List[int]:
        """获取数据完整的有效帧"""
        valid_frames = []
        grouped = concurrent_df.groupby('frame_idx')

        for frame_idx, frame_data in grouped:
            if FrameValidator.is_frame_valid(frame_data):
                valid_frames.append(frame_idx)

        return valid_frames

    def filter_data(self) -> pd.DataFrame:
        """执行数据过滤"""
        print(f"原始数据文件: {self.csv_file_path}")
        print(f"输出文件: {self.output_file_path}")
        print(f"原始数据总行数: {len(self.df)}")
        print(f"原始数据帧数: {self.df['frame_idx'].nunique()}")

        # 第一步：过滤同时存在两个track的帧
        concurrent_df = self._filter_concurrent_frames()
        print(f"同时存在track0和track1的帧数: {len(concurrent_df['frame_idx'].unique())}")
        print(f"第一步过滤后数据行数: {len(concurrent_df)}")

        if len(concurrent_df) == 0:
            print("警告: 没有找到同时存在track0和track1的帧！")
            self.filtered_df = concurrent_df
            return self.filtered_df

        # 第二步：过滤数据完整的帧
        valid_frames = self._get_valid_frames(concurrent_df)
        print(f"完整有效的帧数: {len(valid_frames)}")

        self.filtered_df = concurrent_df[concurrent_df['frame_idx'].isin(valid_frames)].copy()
        return self.filtered_df

    def analyze_results(self) -> None:
        """分析过滤结果"""
        if self.filtered_df is None or len(self.filtered_df) == 0:
            print("没有有效数据可供分析")
            return

        print(f"最终有效数据行数: {len(self.filtered_df)}")
        print(f"最终有效帧数: {self.filtered_df['frame_idx'].nunique()}")
        print(f"总过滤掉行数: {len(self.df) - len(self.filtered_df)}")
        print(f"总过滤掉帧数: {self.df['frame_idx'].nunique() - self.filtered_df['frame_idx'].nunique()}")

        # 验证清洗结果
        print("\n最终各track的数据行数:")
        filtered_track_counts = self.filtered_df['track'].value_counts()
        for track, count in filtered_track_counts.items():
            print(f"  {track}: {count} 行")

        # 检查每帧是否都有两个track
        final_frame_counts = self.filtered_df.groupby('frame_idx')['track'].nunique()
        consistent_frames = len(final_frame_counts[final_frame_counts == 2])
        print(f"完整帧数（每帧都有2个有效track）: {consistent_frames}")

        # 显示帧数范围
        min_frame = self.filtered_df['frame_idx'].min()
        max_frame = self.filtered_df['frame_idx'].max()
        print(f"有效帧数范围: {min_frame} - {max_frame}")

    def save_results(self) -> None:
        """保存过滤结果"""
        if self.filtered_df is not None and len(self.filtered_df) > 0:
            self.filtered_df.to_csv(self.output_file_path, index=False)
            print(f"\n清洗后的数据已保存到: {self.output_file_path}")
        else:
            print("警告: 没有有效数据可保存！")

    def run_pipeline(self) -> Tuple[pd.DataFrame, str]:
        """运行完整的数据处理流程"""
        self.load_data()
        self.filter_data()
        self.analyze_results()
        self.save_results()
        return self.filtered_df, self.output_file_path


def main():
    """
    主函数
    """
    # 只需要指定输入文件，输出文件会自动生成
    input_file = "Top_down_12mm_real.000_TrianOnly_2.analysis.csv"

    # 创建过滤器实例并运行处理流程
    filter_processor = ConcurrentTrackFilter(input_file)
    filtered_data, output_path = filter_processor.run_pipeline()

    # 显示清洗后的数据基本信息
    print(f"\n清洗后数据形状: {filtered_data.shape}")
    if len(filtered_data) > 0:
        print(f"清洗后帧数范围: {filtered_data['frame_idx'].min()} - {filtered_data['frame_idx'].max()}")


if __name__ == "__main__":
    main()