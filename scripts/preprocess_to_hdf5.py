#!/usr/bin/env python
"""HDF5 预处理入口脚本 — 将原始图片数据集转为高速缓存格式。

用法:
    # IP102
    python scripts/preprocess_to_hdf5.py --dataset ip102 --data-dir data/IP102 --output data/cache/ip102.h5

    # PlantVillage
    python scripts/preprocess_to_hdf5.py --dataset plantvillage --data-dir data/PlantVillage --output data/cache/plantvillage.h5

预处理完成后, 训练时加 --cache-dir 参数即可:
    python scripts/train.py --dataset ip102 --model full --cache-dir data/cache/ip102.h5
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.hdf5_dataset import preprocess_to_hdf5


def main():
    parser = argparse.ArgumentParser(
        description="将原始数据集预缓存为 HDF5 格式（消除训练时的I/O瓶颈）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/preprocess_to_hdf5.py --dataset ip102 \\
      --data-dir data/IP102 --output data/cache/ip102.h5

  python scripts/preprocess_to_hdf5.py --dataset plantvillage \\
      --data-dir data/PlantVillage --output data/cache/plantvillage.h5
        """,
    )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ip102", "plantvillage"],
                        help="数据集名称")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="原始数据集根目录")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="输出HDF5文件路径 (建议: data/cache/xxx.h5)")
    parser.add_argument("--target-size", type=int, default=256,
                        help="预处理resize尺寸 (默认256, 应 > 训练input_size)")

    args = parser.parse_args()

    # 执行预处理
    result = preprocess_to_hdf5(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_path=args.output,
        target_size=args.target_size,
    )

    # 输出摘要供后续使用
    print(f"\n📋 后续使用方法:")
    print(f"  python scripts/train.py \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --model full \\")
    print(f"    --cache-dir {args.output}")


if __name__ == "__main__":
    main()
