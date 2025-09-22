import numpy as np
from scipy.spatial import ConvexHull

 def calculate_pose_festures(keypoints, prev_positions=None, frame_time=1/30);
    """
    参数:
    keypoints: 字典，包含各部位坐标 {'nose': [x,y], 'left_ear': [x,y], ...}
    prev_positions: 上一帧的位置字典，用于计算速度 (可选)
    frame_time: 帧时间间隔，默认为1/30秒 (30fps)
    返回:
    features: 字典，包含五个特征值
    """
    # 确保所有关键点都存在
    required_keys = {'nose1','earL1','earR1','tailstart1','tailend1'}