"""
UNet3Plus Training Script
参考UNet++训练代码实现
"""
import os
import argparse
import torch
import time
import sys

from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UNet3Plus
from model import UNet3PlusLoss
from data import AirbusDataset
from utils import SegmentationMetrics, EarlyStopping, TrainingLogger, count_parameters
from tools.common import get_device, set_seed, create_optimizer, create_scheduler, save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="UNet3Plus Training")
    p.add_argument("--data_path", type=str, default=r"D:\resource\data\SS\AirbusShip_filtered_0.01",
                   help="数据集根目录路径")
    p.add_argument("--weight_path", type=str, default="params/unet3p_ship.pth",
                   help="模型权重保存路径")
    p.add_argument("--log_dir", type=str, default="logs",
                   help="日志保存目录")
    p.add_argument("--result_path", type=str, default="result",
                   help="结果保存目录")
    p.add_argument("--batch_size", type=int, default=8,
                   help="训练批次大小")
    p.add_argument("--val_batch_size", type=int, default=4,
                   help="验证批次大小")
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=0.001,
                   help="初始学习率")
    p.add_argument("--num_workers", type=int, default=0,
                   help="数据加载线程数")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                   help="输入图像尺寸 (高 宽)")
    p.add_argument("--smooth", type=float, default=1e-5,
                   help="Dice损失平滑项")
    p.add_argument("--use_augmentation", action="store_true",
                   help="是否使用数据增强")
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="早停机制容忍轮数")
    p.add_argument("--lr_scheduler_patience", type=int, default=5,
                   help="学习率调度器容忍轮数")
    p.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                   help="学习率衰减因子")
    p.add_argument("--resume", type=str, default=None,
                   help="恢复训练的检查点路径")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子")
    p.add_argument("--print_interval", type=int, default=10,
                   help="训练过程打印间隔")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, interval):
    """
    训练一个epoch

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        interval: 打印间隔

    Returns:
        平均损失
    """
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # 前向传播
        preds = model(x)
        loss = criterion(preds, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % interval == 0:
            print(f"  Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader, criterion, device, metrics):
    """
    验证模型

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        device: 设备
        metrics: 指标计算器

    Returns:
        平均损失, 指标字典
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()

            # 计算指标 (UNet3Plus在eval模式下返回单个输出)
            pred_for_metrics = preds if not isinstance(preds, list) else preds[0]
            metrics.update(pred_for_metrics, y)

    return total_loss / len(loader), metrics.get_metrics()


def main():
    """主训练函数"""
    args = parse_args()

    # 设备配置
    device = get_device()
    set_seed(args.seed)

    # 创建目录
    os.makedirs(os.path.dirname(args.weight_path) or ".", exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # 日志配置
    logger = TrainingLogger(log_dir=args.log_dir, log_to_file=True)

    # 创建模型
    model = UNet3Plus(in_channels=3, num_classes=1).to(device)
    params = count_parameters(model)
    logger.model_info("UNet3Plus", params, device)

    # 创建损失函数
    criterion = UNet3PlusLoss(num_classes=1, smooth=args.smooth)

    # 创建数据集
    image_size = tuple(args.image_size)
    train_ds = AirbusDataset(args.data_path, "training", image_size, args.use_augmentation)
    val_ds = AirbusDataset(args.data_path, "validation", image_size, False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.dataset_info(
        len(train_ds),
        len(val_ds),
        args.batch_size,
        args.val_batch_size,
        image_size,
        args.num_workers
    )

    # 优化器和学习率调度器
    optimizer = create_optimizer(model, args.lr)
    scheduler = create_scheduler(optimizer, args.lr_scheduler_patience, args.lr_scheduler_factor)

    # 早停机制
    early_stop = EarlyStopping(args.early_stop_patience, 0.001, "min")

    # 加载检查点
    checkpoint_path = args.resume if args.resume else args.weight_path
    start_epoch, best = load_checkpoint(checkpoint_path, model, optimizer, device)
    best_epoch = 1
    old_lr = args.lr

    print(f"\nTraining start, epochs: {args.epochs}")
    start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.print_interval
        )
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        metrics_tracker = SegmentationMetrics()
        val_loss, metrics = validate(model, val_loader, criterion, device, metrics_tracker)
        print(f"Val Loss: {val_loss:.4f}, IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")

        # 检查是否为最佳模型
        is_best = val_loss < best
        if is_best:
            best, best_epoch = val_loss, epoch
            print(f"*** Best model saved, Val Loss: {best:.4f} ***")

        # 学习率调度
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
        old_lr = new_lr

        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch, val_loss, is_best,
            checkpoint_path, best_val_loss=best
        )

        # 早停检查
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    total = time.time() - start
    print(f"\nTraining complete in {total / 60:.1f} minutes")
    print(f"Best epoch: {best_epoch}, Best Val Loss: {best:.4f}")


if __name__ == "__main__":
    main()
