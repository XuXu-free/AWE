"""
合并阶跃实验和风电实验数据集（多槽版本）
支持从step和wind目录读取多个CSV文件并合并为单个训练数据集
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import glob
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def load_csv_files_from_dir(directory, pattern="*.csv"):
    """
    从指定目录加载所有CSV文件

    Args:
        directory: 数据目录路径
        pattern: 文件名匹配模式

    Returns:
        list: DataFrame列表
    """
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist, skipping...")
        return []

    csv_files = sorted(glob.glob(os.path.join(directory, pattern)))

    if not csv_files:
        print(f"Warning: No CSV files found in {directory} with pattern {pattern}")
        return []

    print(f"Found {len(csv_files)} CSV files in {directory}")

    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                dataframes.append(df)
                print(f"  Loaded: {os.path.basename(csv_file)} ({len(df)} rows)")
            else:
                print(f"  Skipped (empty): {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Error loading {os.path.basename(csv_file)}: {e}")

    return dataframes


def validate_and_align_dataframes(dataframes):
    """
    验证并对齐所有DataFrame的列

    Args:
        dataframes: DataFrame列表

    Returns:
        list: 对齐后的DataFrame列表
    """
    if not dataframes:
        return []

    # 获取所有列的交集
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns &= set(df.columns)

    common_columns = sorted(list(common_columns))

    if len(common_columns) == 0:
        raise ValueError("No common columns found across datasets!")

    print(f"\nCommon columns across all datasets: {len(common_columns)}")

    # 对齐所有DataFrame
    aligned_dfs = []
    for df in dataframes:
        aligned_df = df[common_columns].copy()
        aligned_dfs.append(aligned_df)

    return aligned_dfs


def remove_unnecessary_columns(df, keep_step_time=False):
    """
    删除训练不需要的列

    Args:
        df: 输入DataFrame
        keep_step_time: 是否保留step和time字段

    Returns:
        DataFrame: 处理后的数据
    """
    columns_to_drop = []

    if not keep_step_time:
        # step和time字段在训练中不需要
        if 'step' in df.columns:
            columns_to_drop.append('step')
        if 'time' in df.columns:
            columns_to_drop.append('time')

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"  Dropped columns: {columns_to_drop}")

    return df


def add_missing_prev_actions(df):
    """
    添加缺失的_prev动作字段（如果原始数据中没有）

    Args:
        df: 输入DataFrame

    Returns:
        DataFrame: 处理后的数据
    """
    # 检查并添加I_prev_1~4
    for i in range(1, 5):
        col_name = f'I_prev_{i}'
        if col_name not in df.columns:
            action_col = f'I_{i}'
            if action_col in df.columns:
                df[col_name] = df[action_col].shift(1).fillna(0.0)
                print(f"  Added missing column: {col_name}")

    # 检查并添加v_lye_prev_1~4
    for i in range(1, 5):
        col_name = f'v_lye_prev_{i}'
        if col_name not in df.columns:
            action_col = f'v_lye_{i}'
            if action_col in df.columns:
                df[col_name] = df[action_col].shift(1).fillna(df[action_col].iloc[0] if len(df) > 0 else 0.0)
                print(f"  Added missing column: {col_name}")

    # 检查并添加v_c_prev
    if 'v_c_prev' not in df.columns:
        if 'v_c' in df.columns:
            df['v_c_prev'] = df['v_c'].shift(1).fillna(0.0)
            print("  Added missing column: v_c_prev")

    return df


def merge_datasets(step_dir, wind_dir, root_dir, output_file, keep_step_time=False, validate=True):
    """
    合并step、wind和根目录数据集

    Args:
        step_dir: 阶跃实验数据目录
        wind_dir: 风电实验数据目录
        root_dir: 根目录数据集
        output_file: 输出文件路径
        keep_step_time: 是否保留step和time字段
        validate: 是否验证数据完整性

    Returns:
        bool: 是否成功
    """
    print("=" * 60)
    print("Dataset Merger for Multi-Stack AWE")
    print("=" * 60)

    all_dataframes = []

    # 加载step数据
    if step_dir is not None:
        print(f"\n[1/4] Loading STEP datasets from: {step_dir}")
        step_dfs = load_csv_files_from_dir(step_dir, "nmpc_dataset_step_*.csv")
        if step_dfs:
            all_dataframes.extend(step_dfs)
            print(f"  Total step experiments: {len(step_dfs)}")
    else:
        step_dfs = []
        print("\n[1/4] Skipping STEP datasets (not specified)")

    # 加载wind数据
    if wind_dir is not None:
        print(f"\n[2/4] Loading WIND datasets from: {wind_dir}")
        wind_dfs = load_csv_files_from_dir(wind_dir, "nmpc_dataset_wind_*.csv")
        if wind_dfs:
            all_dataframes.extend(wind_dfs)
            print(f"  Total wind experiments: {len(wind_dfs)}")
    else:
        wind_dfs = []
        print("\n[2/4] Skipping WIND datasets (not specified)")

    # 加载根目录数据
    if root_dir is not None:
        print(f"\n[3/4] Loading ROOT datasets from: {root_dir}")
        root_dfs = load_csv_files_from_dir(root_dir, "nmpc_dataset_*.csv")
        # 过滤掉step和wind目录中的文件（避免重复）
        root_dfs = [df for df in root_dfs if 'step_' not in str(df) and 'wind_' not in str(df)]
        if root_dfs:
            all_dataframes.extend(root_dfs)
            print(f"  Total root experiments: {len(root_dfs)}")
    else:
        root_dfs = []
        print("\n[3/4] Skipping ROOT datasets (not specified)")

    if not all_dataframes:
        print("\nError: No datasets found to merge!")
        return False

    print(f"\nTotal datasets to merge: {len(all_dataframes)}")

    # 验证并对齐列
    print("\n[4/4] Validating and aligning datasets...")
    try:
        aligned_dfs = validate_and_align_dataframes(all_dataframes)
    except ValueError as e:
        print(f"Error: {e}")
        return False

    # 处理每个DataFrame
    processed_dfs = []
    for i, df in enumerate(aligned_dfs):
        print(f"\nProcessing dataset {i+1}/{len(aligned_dfs)}...")

        # 删除不需要的列
        df = remove_unnecessary_columns(df, keep_step_time)

        # 添加缺失的_prev字段
        df = add_missing_prev_actions(df)

        # 删除可能的NaN行（由shift操作产生）
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN values")

        processed_dfs.append(df)

    # 合并所有数据
    print(f"\nMerging {len(processed_dfs)} datasets...")
    merged_df = pd.concat(processed_dfs, ignore_index=True)

    print(f"\nMerged dataset info:")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  Columns: {list(merged_df.columns)[:10]}...")  # 只显示前10个列名

    # 数据验证
    if validate:
        print("\nValidating merged dataset...")

        # 检查必需的训练字段（多槽版本）
        required_fields = [
            'T_s_in', 'T_s_1', 'T_s_2', 'T_s_3', 'T_s_4',
            'T_sep', 'T_c_out', 'n_H2_an_1', 'n_H2_an_2', 'n_H2_an_3', 'n_H2_an_4',
            'n_liq', 'n_gas', 'T_ref',
            'I_prev_1', 'I_prev_2', 'I_prev_3', 'I_prev_4',
            'v_lye_prev_1', 'v_lye_prev_2', 'v_lye_prev_3', 'v_lye_prev_4', 'v_c_prev',
            'I_1', 'I_2', 'I_3', 'I_4',
            'v_lye_1', 'v_lye_2', 'v_lye_3', 'v_lye_4', 'v_c'
        ]

        missing_fields = [f for f in required_fields if f not in merged_df.columns]
        if missing_fields:
            print(f"  Warning: Missing required fields: {missing_fields}")
        else:
            print("  All required training fields present [OK]")

        # 检查plan字段
        plan_fields = [c for c in merged_df.columns if c.startswith('plan_step_')]
        if plan_fields:
            print(f"  Found {len(plan_fields)} plan fields [OK]")
        else:
            print("  Warning: No plan fields found!")

        # 检查是否有NaN
        nan_count = merged_df.isna().sum().sum()
        if nan_count > 0:
            print(f"  Warning: Dataset contains {nan_count} NaN values")
        else:
            print("  No NaN values in dataset [OK]")

    # 保存合并后的数据
    print(f"\nSaving merged dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"  Saved successfully! ({len(merged_df)} rows, {len(merged_df.columns)} columns)")

    # 生成数据摘要
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"Step datasets: {len(step_dfs)}")
    print(f"Wind datasets: {len(wind_dfs)}")
    print(f"Root datasets: {len(root_dfs)}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='合并阶跃实验和风电实验数据集（多槽版本）'
    )
    parser.add_argument('--step_dir', type=str, default=None,
                        help='阶跃实验数据目录 (默认: output/multi_stack/dataset/step/)')
    parser.add_argument('--wind_dir', type=str, default=None,
                        help='风电实验数据目录 (默认: output/multi_stack/dataset/wind/)')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='根目录数据集路径 (默认: output/multi_stack/dataset/)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (默认: output/multi_stack/dataset/merged/nmpc_dataset_merged_<timestamp>.csv)')
    parser.add_argument('--keep_step_time', action='store_true',
                        help='保留step和time字段 (默认: 删除)')
    parser.add_argument('--no_validate', action='store_true',
                        help='跳过数据验证')
    parser.add_argument('--only_step', action='store_true',
                        help='只合并阶跃数据')
    parser.add_argument('--only_wind', action='store_true',
                        help='只合并风电数据')
    parser.add_argument('--include_root', action='store_true',
                        help='包含根目录的数据集')

    args = parser.parse_args()

    # 设置默认路径
    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'output', 'multi_stack', 'dataset'
    ))

    if args.step_dir is None:
        step_dir = os.path.join(base_dir, 'step')
    else:
        step_dir = args.step_dir

    if args.wind_dir is None:
        wind_dir = os.path.join(base_dir, 'wind')
    else:
        wind_dir = args.wind_dir

    if args.root_dir is None:
        root_dir = base_dir
    else:
        root_dir = args.root_dir

    # 如果只合并一种数据
    if args.only_step:
        wind_dir = None
        root_dir = None
    if args.only_wind:
        step_dir = None
        root_dir = None

    # 设置默认输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f'merged_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        if args.only_step:
            output_file = os.path.join(output_dir, f'nmpc_dataset_merged_step_{timestamp}.csv')
        elif args.only_wind:
            output_file = os.path.join(output_dir, f'nmpc_dataset_merged_wind_{timestamp}.csv')
        else:
            output_file = os.path.join(output_dir, f'nmpc_dataset_merged_{timestamp}.csv')
    else:
        output_file = args.output

    # 执行合并
    success = merge_datasets(
        step_dir=step_dir,
        wind_dir=wind_dir,
        root_dir=root_dir if args.include_root else None,
        output_file=output_file,
        keep_step_time=args.keep_step_time,
        validate=not args.no_validate
    )

    if success:
        print("\n[OK] Dataset merging completed successfully!")
    else:
        print("\n[FAILED] Dataset merging failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
