"""
过滤 Airbus Ship 数据集
1. 过滤出有船的图片
2. 按2:8划分验证集和训练集
3. 可配置采样比例
4. 支持输出图像尺寸缩放
"""
import os
import shutil
import random
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Airbus Ship 数据集过滤工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 路径配置
    parser.add_argument('--source_data_root', type=str,
                       default=r"D:\resource\data\OD\AirbusShip",
                       help='源数据集根目录')
    parser.add_argument('--target_data_root', type=str,
                       default=r"D:\resource\data\SS\AirbusShip_filtered_0.2",
                       help='目标数据集根目录')
    parser.add_argument('--source_img_dir', type=str, default="train_v2",
                       help='源图像目录名')
    parser.add_argument('--source_csv', type=str,
                       default="train_ship_segmentations_v2.csv",
                       help='源标注CSV文件名')

    # 数据集参数
    parser.add_argument('--sample_ratio', type=float, default=0.2,
                       help='采样比例 (0.0 - 1.0)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集划分比例 (0.0 - 1.0)')
    parser.add_argument('--image_size', type=int, nargs=2,
                       default=(256, 256),
                       help='输出图像尺寸 (高度 宽度)')

    # 其他参数
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--overwrite', action='store_true',
                       help='强制覆盖现有数据集')

    return parser.parse_args()


def rle_decode(mask_rle, shape=(768, 768)):
    """将RLE编码解码为掩码图像"""
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape).T


def main():
    args = parse_args()

    print("=" * 60)
    print("Airbus Ship 数据集过滤工具")
    print("=" * 60)

    random.seed(args.random_seed)

    # 路径配置
    SOURCE_IMG_DIR = os.path.join(args.source_data_root, args.source_img_dir)
    SOURCE_CSV = os.path.join(args.source_data_root, args.source_csv)
    TARGET_DATA_ROOT = args.target_data_root
    OUTPUT_SIZE = tuple(args.image_size)

    # 检查源路径
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"❌ 源图像目录不存在: {SOURCE_IMG_DIR}")
        return
    if not os.path.exists(SOURCE_CSV):
        print(f"❌ 源标注文件不存在: {SOURCE_CSV}")
        return

    # 处理目标目录
    if os.path.exists(TARGET_DATA_ROOT):
        if args.overwrite:
            print(f"⚠️  目标目录已存在，将强制覆盖: {TARGET_DATA_ROOT}")
            shutil.rmtree(TARGET_DATA_ROOT)
        else:
            print(f"⚠️  目标目录已存在，请使用 --overwrite 强制覆盖")
            return

    # 创建目标目录结构
    dirs = [
        os.path.join(TARGET_DATA_ROOT, "images", "training"),
        os.path.join(TARGET_DATA_ROOT, "images", "validation"),
        os.path.join(TARGET_DATA_ROOT, "annotations", "training"),
        os.path.join(TARGET_DATA_ROOT, "annotations", "validation"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"\n【配置】")
    print(f"源数据: {args.source_data_root}")
    print(f"目标数据: {TARGET_DATA_ROOT}")
    print(f"图像尺寸: {OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}")
    print(f"采样比例: {args.sample_ratio*100:.0f}%")
    print(f"验证集比例: {args.val_split*100:.0f}%")

    # 1. 读取CSV文件
    print(f"\n【1. 读取标注文件】")
    df = pd.read_csv(SOURCE_CSV)
    print(f"总标注记录数: {len(df)}")

    # 2. 过滤出有船的图片
    print(f"\n【2. 过滤有船的图片】")

    df_ships = df[df['EncodedPixels'].notna()].copy()
    ship_counts = df_ships.groupby('ImageId').size()

    all_images = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith('.jpg')]
    print(f"原始图片总数: {len(all_images)}")

    # 找出有船的图片
    valid_images = []
    for img_name in tqdm(all_images, desc="过滤有船图片"):
        if img_name in ship_counts.index:
            valid_images.append(img_name)

    print(f"有船的图片: {len(valid_images)} ({len(valid_images)/len(all_images)*100:.1f}%)")
    print(f"空白图片: {len(all_images) - len(valid_images)} ({(len(all_images) - len(valid_images))/len(all_images)*100:.1f}%)")

    # 3. 划分验证集和训练集
    print(f"\n【3. 划分训练集和验证集】")
    print(f"验证集: {args.val_split*100:.0f}%")
    print(f"训练集: {100 - args.val_split*100:.0f}%")

    random.shuffle(valid_images)

    val_split_idx = int(len(valid_images) * args.val_split)
    val_images_full = valid_images[:val_split_idx]
    train_images_full = valid_images[val_split_idx:]

    print(f"验证集(全部): {len(val_images_full)} 张")
    print(f"训练集(全部): {len(train_images_full)} 张")

    # 4. 采样
    print(f"\n【4. 采样 {args.sample_ratio*100:.0f}% 作为最终数据集】")

    random.shuffle(val_images_full)
    random.shuffle(train_images_full)

    val_size = int(len(val_images_full) * args.sample_ratio)
    train_size = int(len(train_images_full) * args.sample_ratio)

    val_images = val_images_full[:val_size]
    train_images = train_images_full[:train_size]

    print(f"最终验证集: {len(val_images)} 张 (验证集{len(val_images)/len(val_images_full)*100:.0f}%，整体{len(val_images)/len(valid_images)*100:.1f}%)")
    print(f"最终训练集: {len(train_images)} 张 (训练集{len(train_images)/len(train_images_full)*100:.0f}%，整体{len(train_images)/len(valid_images)*100:.1f}%)")
    print(f"总计: {len(train_images) + len(val_images)} 张")

    # 5. 生成数据集
    print(f"\n【5. 生成掩码并复制文件】")
    print(f"输出图像尺寸: {OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}")

    # 处理训练集
    print(f"\n处理训练集...")
    for img_name in tqdm(train_images, desc="训练集"):
        # 复制并缩放图像
        src = os.path.join(SOURCE_IMG_DIR, img_name)
        dst = os.path.join(TARGET_DATA_ROOT, "images", "training", img_name)

        image = Image.open(src).convert('RGB')
        image_resized = image.resize(OUTPUT_SIZE, Image.BILINEAR)
        image_resized.save(dst)

        # 生成并缩放掩码
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(TARGET_DATA_ROOT, "annotations", "training", mask_name)

        image_rles = df[df['ImageId'] == img_name]['EncodedPixels'].values
        mask = np.zeros((768, 768), dtype=np.uint8)

        for rle in image_rles:
            if pd.notna(rle):
                ship_mask = rle_decode(rle, shape=(768, 768))
                mask = np.maximum(mask, ship_mask)

        mask_img = Image.fromarray(mask * 255).resize(OUTPUT_SIZE, Image.NEAREST)
        mask_img.save(mask_path)

    # 处理验证集
    print(f"\n处理验证集...")
    for img_name in tqdm(val_images, desc="验证集"):
        # 复制并缩放图像
        src = os.path.join(SOURCE_IMG_DIR, img_name)
        dst = os.path.join(TARGET_DATA_ROOT, "images", "validation", img_name)

        image = Image.open(src).convert('RGB')
        image_resized = image.resize(OUTPUT_SIZE, Image.BILINEAR)
        image_resized.save(dst)

        # 生成并缩放掩码
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(TARGET_DATA_ROOT, "annotations", "validation", mask_name)

        image_rles = df[df['ImageId'] == img_name]['EncodedPixels'].values
        mask = np.zeros((768, 768), dtype=np.uint8)

        for rle in image_rles:
            if pd.notna(rle):
                ship_mask = rle_decode(rle, shape=(768, 768))
                mask = np.maximum(mask, ship_mask)

        mask_img = Image.fromarray(mask * 255).resize(OUTPUT_SIZE, Image.NEAREST)
        mask_img.save(mask_path)

    # 6. 完成
    print(f"\n【6. 完成】")
    print(f"=" * 60)
    print(f"数据集生成完成!")
    print(f"数据集路径: {TARGET_DATA_ROOT}")
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    print(f"总计: {len(train_images) + len(val_images)} 张")
    print(f"=" * 60)

    print(f"\n目录结构:")
    print(f"{TARGET_DATA_ROOT}/")
    print(f"├── images/")
    print(f"│   ├── training/      ({len(train_images)} 张)")
    print(f"│   └── validation/    ({len(val_images)} 张)")
    print(f"└── annotations/")
    print(f"    ├── training/      ({len(train_images)} 张)")
    print(f"    └── validation/    ({len(val_images)} 张)")

    print(f"\n下一步:")
    print(f"修改训练脚本参数以匹配新数据集")


if __name__ == '__main__':
    main()
