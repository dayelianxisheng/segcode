"""
TransUNet评估脚本
"""
import os
import argparse
import torch
import time
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import create_transunet, CombinedLoss
from data import AirbusDataset
from utils import SegmentationMetrics
from tools.common import get_device


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="TransUNet Evaluation")
    p.add_argument("--data_path", type=str, required=True,
                   help="数据集路径")
    p.add_argument("--weight_path", type=str, default="params/transunet_ship.pth",
                   help="模型权重路径")
    p.add_argument("--variant", type=str, default="vit_b16",
                   choices=["vit_b16", "vit_b32", "vit_l16"],
                   help="TransUNet变体")
    p.add_argument("--img_size", type=int, default=224,
                   help="输入图像尺寸")
    p.add_argument("--batch_size", type=int, default=4,
                   help="批次大小")
    p.add_argument("--num_workers", type=int, default=0,
                   help="数据加载线程数")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="二值化阈值")
    p.add_argument("--save_predictions", action="store_true",
                   help="是否保存预测结果")
    p.add_argument("--output_dir", type=str, default="result/transunet_eval",
                   help="预测结果保存目录")
    return p.parse_args()


def evaluate(model, loader, device, threshold=0.5, save_dir=None):
    """评估模型"""
    model.eval()
    metrics_tracker = SegmentationMetrics()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_time = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # 计时
            start = time.time()
            outputs = model(x)
            elapsed = time.time() - start
            total_time += elapsed

            # 更新指标
            metrics_tracker.update(outputs, y, threshold=threshold)

            # 保存预测结果
            if save_dir and i == 0:
                import cv2
                import numpy as np
                for j in range(min(4, x.size(0))):
                    pred_mask = (outputs[j].squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
                    gt_mask = (y[j].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

                    cv2.imwrite(os.path.join(save_dir, f"pred_{i}_{j}.png"), pred_mask)
                    cv2.imwrite(os.path.join(save_dir, f"gt_{i}_{j}.png"), gt_mask)

    # 计算平均指标
    metrics = metrics_tracker.get_metrics()
    avg_time = total_time / len(loader)

    return metrics, avg_time


def main():
    """主函数"""
    args = parse_args()

    # 设备
    device = get_device()

    # 创建模型
    print(f"加载TransUNet模型: {args.variant}")
    model = create_transunet(
        variant=args.variant,
        img_size=args.img_size,
        in_channels=3,
        num_classes=1
    ).to(device)

    # 加载权重
    if os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从epoch {checkpoint.get('epoch', 'unknown')}加载权重")
            print(f"验证损失: {checkpoint.get('val_loss', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        print(f"模型权重已加载: {args.weight_path}")
    else:
        print(f"警告: 权重文件不存在: {args.weight_path}")
        return

    # 数据集
    image_size = (args.img_size, args.img_size)
    val_ds = AirbusDataset(args.data_path, "validation", image_size, False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"验证集大小: {len(val_ds)}")

    # 评估
    print("\n开始评估...")
    metrics, avg_time = evaluate(
        model,
        val_loader,
        device,
        threshold=args.threshold,
        save_dir=args.output_dir if args.save_predictions else None
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"IoU:        {metrics['iou']:.4f}")
    print(f"Dice:       {metrics['dice']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"平均推理时间: {avg_time:.4f}秒/batch")
    print("=" * 50)

    if args.save_predictions:
        print(f"\n预测结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
