import os
import time
from typing import Dict, Any


def main_analysis_pipeline():
    """
    完整的动物行为数据分析流水线
    """
    print("=" * 60)
    print("动物行为数据分析流水线")
    print("=" * 60)

    # 记录开始时间
    start_time = time.time()

    # 1. 数据清洗和预处理
    print("\n" + "=" * 50)
    print("阶段1: 数据清洗和预处理")
    print("=" * 50)

    from datawash import ConcurrentTrackFilter

    input_file = "Top_down_12mm_real.000_TrianOnly_2.analysis.csv"
    cleaner = ConcurrentTrackFilter(input_file)
    cleaned_data, cleaned_output_path = cleaner.run_pipeline()

    if cleaned_data.empty:
        print("数据清洗失败，请检查输入文件")
        return

    # 2. 特征计算和提取
    print("\n" + "=" * 50)
    print("阶段2: 特征计算和提取")
    print("=" * 50)

    from Feature import MultiTrackAnalyzer

    feature_analyzer = MultiTrackAnalyzer()
    feature_analyzer.load_data(cleaned_output_path)
    feature_results = feature_analyzer.analyze_all_tracks()
    feature_analyzer.save_all_tracks()

    # 3. 数据归一化
    print("\n" + "=" * 50)
    print("阶段3: 数据归一化")
    print("=" * 50)

    from Normalization import MouseTrackNormalizer, MouseTrackVisualizer

    # 归一化处理
    normalizer = MouseTrackNormalizer()
    track_files = normalizer.find_track_files('track*_processed.csv')

    if not track_files:
        print("未找到特征文件，请检查阶段2的输出")
        return

    normalized_dfs = normalizer.normalize_all_tracks()

    # 可视化
    visualizer = MouseTrackVisualizer()

    # 检查可用特征
    if normalized_dfs:
        first_df = list(normalized_dfs.values())[0]
        available_features = [col for col in first_df.columns if col not in ['track', 'frame_idx']]

        # 绘制关键特征图
        for feature in ['speed', 'acceleration', 'body_length']:
            if feature in available_features:
                visualizer.plot_normalized_features(
                    normalized_dfs,
                    feature_name=feature,
                    save_path=f'normalized_{feature}_comparison.png'
                )

    # 4. PCA分析
    print("\n" + "=" * 50)
    print("阶段4: PCA降维分析")
    print("=" * 50)

    from PCA_combin import PCAAnalyzer

    pca_analyzer = PCAAnalyzer()

    # 自动PCA分析
    print("执行自动PCA分析...")
    auto_results = pca_analyzer.main_comparison_auto("pca_auto_results")

    # 手动筛选PCA分析
    print("执行手动筛选PCA分析...")
    manual_results = pca_analyzer.main_comparison_manual("pca_manual_results")

    print("\n生成PCA自动和手动对比柱状图...")
    if auto_results is not None and manual_results is not None:
        # 获取自动和手动PCA的特征矩阵
        auto_matrix = auto_results['combined_matrix']
        manual_matrix = manual_results['combined_matrix']

        # 生成对比图
        comparison_results = pca_analyzer.plot_dimension_comparison(
            auto_matrix,
            manual_matrix,
            "pca_auto_vs_manual_comparison.png"
        )

        print(f"自动PCA达到90%方差所需维度: {comparison_results['auto_components']}")
        print(f"手动PCA达到90%方差所需维度: {comparison_results['manual_components']}")
        print("PCA对比图已保存为: pca_auto_vs_manual_comparison.png")

    # 5. CCA关联分析
    print("\n" + "=" * 50)
    print("阶段5: CCA关联分析")
    print("=" * 50)

    from CCA import CCAAnalyzer

    cca_analyzer = CCAAnalyzer(output_dir="cca_analysis_results")

    # 检查归一化文件是否存在
    normalized_files = [
        "track_0_processed_normalized.csv",
        "track_1_processed_normalized.csv"
    ]

    # 如果文件不存在，尝试查找其他可能的名字
    if not all(os.path.exists(f) for f in normalized_files):
        # 尝试查找实际的文件名
        actual_files = []
        for pattern in ['track0*normalized.csv', 'track1*normalized.csv', 'track_0*normalized.csv',
                        'track_1*normalized.csv']:
            found_files = glob.glob(pattern)
            actual_files.extend(found_files)

        if len(actual_files) >= 2:
            normalized_files = sorted(actual_files)[:2]  # 取前两个文件
            print(f"使用找到的文件: {normalized_files}")

    if all(os.path.exists(f) for f in normalized_files):
        cca_results = cca_analyzer.analyze(
            file_paths=normalized_files,
            group_names=["Track0", "Track1"],
            top_k=5
        )
    else:
        print("警告: 未找到归一化文件，跳过CCA分析")
        cca_results = None

    # 6. 生成最终报告
    print("\n" + "=" * 50)
    print("阶段6: 生成分析报告")
    print("=" * 50)

    generate_summary_report({
        'cleaned_data': cleaned_data,
        'feature_results': feature_results,
        'normalized_dfs': normalized_dfs,
        'pca_auto': auto_results,
        'pca_manual': manual_results,
        'cca_results': cca_results,
        'processing_time': time.time() - start_time
    })

    print("\n" + "=" * 60)
    print("分析流水线完成！")
    print("=" * 60)
    print(f"总处理时间: {time.time() - start_time:.2f} 秒")
    print("所有结果已保存到相应目录中")


def generate_summary_report(results: Dict[str, Any]):
    """
    生成分析摘要报告

    Parameters:
    -----------
    results : dict
        包含各阶段分析结果的字典
    """
    report_path = "analysis_summary_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("动物行为数据分析摘要报告\n")
        f.write("=" * 60 + "\n\n")

        # 数据清洗结果
        f.write("1. 数据清洗结果:\n")
        f.write("-" * 40 + "\n")
        if 'cleaned_data' in results and results['cleaned_data'] is not None:
            f.write(f"原始数据行数: {results['cleaned_data'].shape[0] if not results['cleaned_data'].empty else 0}\n")
            f.write(
                f"有效帧数: {results['cleaned_data']['frame_idx'].nunique() if not results['cleaned_data'].empty else 0}\n")
        f.write("\n")

        # 特征提取结果
        f.write("2. 特征提取结果:\n")
        f.write("-" * 40 + "\n")
        if 'feature_results' in results and results['feature_results']:
            for track_name, features in results['feature_results'].items():
                f.write(f"{track_name}: {features.get('total_frames', 0)} 帧\n")
        f.write("\n")

        # PCA分析结果
        f.write("3. PCA分析结果:\n")
        f.write("-" * 40 + "\n")
        if 'pca_auto' in results and results['pca_auto']:
            pca_result = results['pca_auto']['pca_full_result']
            f.write(f"前5个主成分累积方差: {sum(pca_result['explained_variance_ratio'][:5]):.3f}\n")
        f.write("\n")

        # CCA分析结果
        f.write("4. CCA关联分析结果:\n")
        f.write("-" * 40 + "\n")
        if 'cca_results' in results and results['cca_results']:
            common_features = results['cca_results']['common_features']
            if 'feature_score_pairs' in common_features:
                f.write("主要共性特征:\n")
                for name, score in common_features['feature_score_pairs']:
                    f.write(f"  - {name}: {score:.4f}\n")
        f.write("\n")

        # 处理时间
        f.write("5. 处理统计:\n")
        f.write("-" * 40 + "\n")
        f.write(f"总处理时间: {results.get('processing_time', 0):.2f} 秒\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("分析完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 60 + "\n")

    print(f"分析报告已保存到: {report_path}")


def run_individual_stage(stage_name: str):
    """
    单独运行某个分析阶段

    Parameters:
    -----------
    stage_name : str
        阶段名称: 'clean', 'feature', 'normalize', 'pca', 'cca'
    """
    if stage_name == 'clean':
        from data_cleaning import ConcurrentTrackFilter
        input_file = "Top_down_12mm_real.000_TrianOnly_2.analysis.csv"
        cleaner = ConcurrentTrackFilter(input_file)
        cleaner.run_pipeline()

    elif stage_name == 'feature':
        from feature_extraction import MultiTrackAnalyzer
        analyzer = MultiTrackAnalyzer()
        analyzer.load_data("Top_down_12mm_real.000_TrianOnly_2.analysis_washed.csv")
        analyzer.analyze_all_tracks()
        analyzer.save_all_tracks()

    elif stage_name == 'normalize':
        from data_normalization import MouseTrackNormalizer, MouseTrackVisualizer
        normalizer = MouseTrackNormalizer()
        normalizer.find_track_files('track*_processed.csv')
        normalized_dfs = normalizer.normalize_all_tracks()

    elif stage_name == 'pca':
        from pca_analysis import PCAAnalyzer
        analyzer = PCAAnalyzer()
        analyzer.main_comparison_auto("pca_auto_results")
        analyzer.main_comparison_manual("pca_manual_results")

    elif stage_name == 'cca':
        from cca_analysis import CCAAnalyzer
        analyzer = CCAAnalyzer(output_dir="cca_analysis_results")
        analyzer.analyze(
            file_paths=["track_0_processed_normalized.csv", "track_1_processed_normalized.csv"],
            group_names=["Track0", "Track1"],
            top_k=5
        )

    else:
        print("无效的阶段名称，可用选项: 'clean', 'feature', 'normalize', 'pca', 'cca'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='动物行为数据分析流水线')
    parser.add_argument('--stage', type=str, help='运行特定阶段: clean, feature, normalize, pca, cca')
    parser.add_argument('--full', action='store_true', help='运行完整流水线')

    args = parser.parse_args()

    if args.stage:
        print(f"运行单个阶段: {args.stage}")
        run_individual_stage(args.stage)
    elif args.full:
        print("运行完整分析流水线")
        main_analysis_pipeline()
    else:
        # 默认运行完整流水线
        print("运行完整分析流水线")
        main_analysis_pipeline()