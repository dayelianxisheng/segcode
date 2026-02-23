"""
UNet3Plus Evaluation Script
在验证集上评估模型性能
"""
import os
import argparse
import torch
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UNet3Plus
from model import UNet3PlusLoss
from data import AirbusDataset
from utils import SegmentationMetrics, count_parameters
from tools.common import get_device


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="UNet3Plus Evaluation")
    p.add_argument("--data_path", type=str,
                   default=r"F:\resource\data\airbusship\AirbusShip_filtered",
                   help="数据集根目录路径")
    p.add_argument("--weight_path", type=str, default="params/unet3p_ship_best.pth",
                   help="模型权重路径")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                   help="输入图像尺寸 (高 宽)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="批次大小")
    p.add_argument("--num_workers", type=int, default=0,
                   help="数据加载线程数")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="二值化阈值")
    return p.parse_args()


def evaluate(model, loader, criterion, device, threshold=0.5):
    """
    评估模型

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        device: 设备
        threshold: 二值化阈值

    Returns:
        平均损失, 指标字典
    """
    model.eval()
    total_loss = 0
    metrics_tracker = SegmentationMetrics()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()

            # 计算指标
            pred_for_metrics = preds if not isinstance(preds, list) else preds[0]
            metrics_tracker.update(pred_for_metrics, y, threshold=threshold)

    avg_loss = total_loss / len(loader)
    metrics = metrics_tracker.get_metrics()

    return avg_loss, metrics


def main():
    """主评估函数"""
    args = parse_args()

    # 设备配置
    device = get_device()

    print("=" * 60)
    print("UNet3Plus 模型评估")
    print("=" * 60)

    # 创建模型
    model = UNet3Plus(in_channels=3, num_classes=1).to(device)
    params = count_parameters(model)
    print(f"模型参数量: {params:,}")
    print(f"设备: {device}")
    print(f"权重路径: {args.weight_path}")

    # 加载权重
    if not os.path.exists(args.weight_path):
        print(f"错误: 权重文件不存在: {args.weight_path}")
        return

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载权重 (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 创建数据集
    image_size = tuple(args.image_size)
    val_ds = AirbusDataset(args.data_path, "validation", image_size, False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"验证集大小: {len(val_ds)}")
    print(f"批次大小: {args.batch_size}")
    print(f"阈值: {args.threshold}")
    print("=" * 60)

    # 创建损失函数
    criterion = UNet3PlusLoss(num_classes=1)

    # 评估
    print("\n开始评估...")
    val_loss, metrics = evaluate(model, val_loader, criterion, device, args.threshold)

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"验证损失: {val_loss:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
