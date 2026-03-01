"""
TransUNet训练脚本
"""
import os
import argparse
import torch
import time

from torch import nn, optim
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import create_transunet, CombinedLoss
from data import AirbusDataset
from utils import SegmentationMetrics, EarlyStopping, TrainingLogger, count_parameters
from tools.common import get_device, set_seed, create_optimizer, create_scheduler, save_checkpoint, load_checkpoint


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="TransUNet Training")
    p.add_argument("--data_path", type=str, default=r"D:\resource\data\seg\AirbusShip_filtered_0.5",
                   help="数据集路径")
    p.add_argument("--weight_path", type=str, default="params/transunet_ship.pth",
                   help="模型权重保存路径")
    p.add_argument("--log_dir", type=str, default="logs",
                   help="日志保存目录")
    p.add_argument("--result_path", type=str, default="result",
                   help="结果保存目录")
    p.add_argument("--variant", type=str, default="vit_b16",
                   choices=["vit_b16", "vit_b32", "vit_l16"],
                   help="TransUNet变体")
    p.add_argument("--img_size", type=int, default=256,
                   help="输入图像尺寸")
    p.add_argument("--batch_size", type=int, default=8,
                   help="训练批次大小")
    p.add_argument("--val_batch_size", type=int, default=4,
                   help="验证批次大小")
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=0.0001,
                   help="学习率")
    p.add_argument("--num_workers", type=int, default=0,
                   help="数据加载线程数")
    p.add_argument("--bce_weight", type=float, default=0.5,
                   help="BCE损失权重")
    p.add_argument("--dice_weight", type=float, default=0.5,
                   help="Dice损失权重")
    p.add_argument("--use_augmentation", action="store_true",
                   help="是否使用数据增强")
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="早停耐心值")
    p.add_argument("--lr_scheduler_patience", type=int, default=5,
                   help="学习率调度器耐心值")
    p.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                   help="学习率衰减因子")
    p.add_argument("--resume", type=str, default=None,
                   help="恢复训练的检查点路径")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子")
    p.add_argument("--print_interval", type=int, default=10,
                   help="打印间隔")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, interval):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % interval == 0:
            print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader, criterion, device, metrics):
    """验证模型"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            metrics.update(outputs, y)

    avg_loss = total_loss / len(loader)
    return avg_loss, metrics.get_metrics()


def main():
    """主函数"""
    args = parse_args()

    # 设备配置
    device = get_device()
    set_seed(args.seed)

    # 创建目录
    os.makedirs(os.path.dirname(args.weight_path) or ".", exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # 日志
    logger = TrainingLogger(log_dir=args.log_dir, log_to_file=True)

    # 创建模型
    print(f"创建TransUNet模型: {args.variant}")
    model = create_transunet(
        variant=args.variant,
        img_size=args.img_size,
        in_channels=3,
        num_classes=1
    ).to(device)

    params = count_parameters(model)
    logger.model_info(f"TransUNet-{args.variant}", params, device)

    # 损失函数
    criterion = CombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)

    # 数据集
    image_size = (args.img_size, args.img_size)
    train_ds = AirbusDataset(args.data_path, "training", image_size, args.use_augmentation)
    val_ds = AirbusDataset(args.data_path, "validation", image_size, False)

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
    early_stop = EarlyStopping(args.early_stop_patience, 0.001, "min")

    # 检查点
    checkpoint_path = args.resume if args.resume else args.weight_path
    start_epoch, best = load_checkpoint(checkpoint_path, model, optimizer, device)
    best_epoch = 1
    old_lr = args.lr

    print(f"Training start, epochs: {args.epochs}")
    start_time = time.time()

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.print_interval)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        metrics_tracker = SegmentationMetrics()
        val_loss, metrics = validate(model, val_loader, criterion, device, metrics_tracker)
        print(f"Val Loss: {val_loss:.4f}, IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")

        # 记录最佳模型
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
        save_checkpoint(model, optimizer, epoch, val_loss, is_best, checkpoint_path, best_val_loss=best)

        # 早停
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch}, Best Val Loss: {best:.4f}")


if __name__ == "__main__":
    main()
