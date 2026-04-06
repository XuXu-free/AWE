"""
整理现有数据集到时间戳子文件夹中
"""
import os
import shutil
import glob
from datetime import datetime

def get_timestamp_from_filename(filename):
    """从文件名中提取时间戳 (YYYYMMDD_HHMMSS)"""
    import re
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return None

def organize_files_by_timestamp(base_dir, dry_run=True):
    """将文件按时间戳组织到子文件夹中"""
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return

    # 获取所有文件（不包括子目录）
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

    # 按时间戳分组
    groups = {}
    for f in files:
        ts = get_timestamp_from_filename(f)
        if ts:
            if ts not in groups:
                groups[ts] = []
            groups[ts].append(f)

    # 移动文件
    for ts, file_list in groups.items():
        target_dir = os.path.join(base_dir, ts)
        if dry_run:
            print(f"[DRY RUN] 将创建: {target_dir}")
            for f in file_list:
                print(f"  移动: {f}")
        else:
            os.makedirs(target_dir, exist_ok=True)
            for f in file_list:
                src = os.path.join(base_dir, f)
                dst = os.path.join(target_dir, f)
                shutil.move(src, dst)
                print(f"  已移动: {f} -> {target_dir}/")

def organize_subfolder(base_dir, subfolder_name, target_prefix, dry_run=True):
    """将子文件夹中的文件按时间戳组织到新文件夹中"""
    subfolder = os.path.join(base_dir, subfolder_name)
    if not os.path.exists(subfolder):
        print(f"子文件夹不存在: {subfolder}")
        return

    # 获取所有文件
    files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
    if not files:
        print(f"子文件夹为空: {subfolder}")
        return

    # 获取时间戳（从第一个文件）
    ts = get_timestamp_from_filename(files[0])
    if not ts:
        print(f"无法从文件名提取时间戳: {files[0]}")
        return

    # 目标文件夹名
    target_dir_name = f"{target_prefix}_{ts}"
    target_dir = os.path.join(base_dir, target_dir_name)

    if dry_run:
        print(f"[DRY RUN] 将创建: {target_dir}")
        for f in files:
            print(f"  移动: {subfolder}/{f} -> {target_dir}/")
    else:
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            src = os.path.join(subfolder, f)
            dst = os.path.join(target_dir, f)
            shutil.move(src, dst)
        print(f"  已移动 {len(files)} 个文件: {subfolder} -> {target_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Organize datasets into timestamped folders')
    parser.add_argument('--execute', action='store_true', help='实际执行移动操作（默认仅预览）')
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("="*60)
        print("预览模式 (DRY RUN) - 添加 --execute 参数实际执行")
        print("="*60)
    else:
        print("="*60)
        print("执行模式 - 正在移动文件")
        print("="*60)

    # Multi-stack
    print("\n### Multi-stack Dataset ###")
    ms_base = "output/multi_stack/dataset"

    # 1. 整理根目录文件
    print("\n整理根目录文件...")
    organize_files_by_timestamp(ms_base, dry_run)

    # 2. 整理 step 子文件夹
    print("\n整理 step 子文件夹...")
    organize_subfolder(ms_base, "step", "step", dry_run)

    # 3. 整理 wind 子文件夹
    print("\n整理 wind 子文件夹...")
    organize_subfolder(ms_base, "wind", "wind", dry_run)

    # 4. 整理 merged 子文件夹
    print("\n整理 merged 子文件夹...")
    organize_subfolder(ms_base, "merged", "merged", dry_run)

    # Single-stack
    print("\n### Single-stack Dataset ###")
    ss_base = "output/single_stack/dataset"

    # 1. 整理根目录文件
    print("\n整理根目录文件...")
    organize_files_by_timestamp(ss_base, dry_run)

    # 2. 整理 step 子文件夹
    print("\n整理 step 子文件夹...")
    organize_subfolder(ss_base, "step", "step", dry_run)

    # 3. 整理 wind 子文件夹
    print("\n整理 wind 子文件夹...")
    organize_subfolder(ss_base, "wind", "wind", dry_run)

    # 4. 整理 merged 子文件夹
    print("\n整理 merged 子文件夹...")
    organize_subfolder(ss_base, "merged", "merged", dry_run)

    print("\n" + "="*60)
    if dry_run:
        print("预览完成。运行以下命令实际执行:")
        print("  python scripts/organize_datasets.py --execute")
    else:
        print("整理完成!")
    print("="*60)

if __name__ == "__main__":
    main()
