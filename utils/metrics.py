"""
评估指标计算
"""
import torch
import numpy as np


def calculate_iou(pred, target):
    """
    计算IoU (Intersection over Union)

    Args:
        pred: 预测值 (B, H, W) 或 (B, 1, H, W)
        target: 真实标签 (B, H, W) 或 (B, 1, H, W)

    Returns:
        iou: IoU值
    """
    # 确保形状一致
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # 计算TP, FP, FN
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    # IoU
    iou = tp / (tp + fp + fn + 1e-8)

    return iou.item()


def calculate_dice(pred, target):
    """
    计算Dice系数

    Dice = 2*TP / (2*TP + FP + FN)
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    return dice.item()


def calculate_precision_recall(pred, target):
    """
    计算精确率和召回率

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return precision.item(), recall.item()


def calculate_metrics(pred, target, threshold=0.5):
    """
    计算所有指标

    Returns:
        dict: {'iou', 'dice', 'precision', 'recall'}
    """
    # 二值化
    pred_binary = (pred > threshold).float()

    iou = calculate_iou(pred_binary, target)
    dice = calculate_dice(pred_binary, target)
    precision, recall = calculate_precision_recall(pred_binary, target)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }


class SegmentationMetrics:
    """分割指标追踪器"""
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.iou_scores = []
        self.dice_scores = []
        self.precision_scores = []
        self.recall_scores = []

    def update(self, pred, target, threshold=0.5):
        """更新指标"""
        metrics = calculate_metrics(pred, target, threshold)

        self.iou_scores.append(metrics['iou'])
        self.dice_scores.append(metrics['dice'])
        self.precision_scores.append(metrics['precision'])
        self.recall_scores.append(metrics['recall'])

    def get_metrics(self):
        """获取平均指标"""
        return {
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'precision': np.mean(self.precision_scores) if self.precision_scores else 0.0,
            'recall': np.mean(self.recall_scores) if self.recall_scores else 0.0
        }

    def print_metrics(self):
        """打印指标"""
        metrics = self.get_metrics()
        print(f"IoU: {metrics['iou']:.4f}, "
              f"Dice: {metrics['dice']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}")


if __name__ == '__main__':
    # 测试指标计算
    print("=" * 60)
    print("测试评估指标")
    print("=" * 60)

    # 创建模拟数据
    pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()

    # 测试单个batch
    metrics = calculate_metrics(pred, target)
    print(f"单个batch指标:")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    # 测试指标追踪器
    print("\n指标追踪器测试:")
    tracker = SegmentationMetrics()
    for i in range(5):
        pred = torch.sigmoid(torch.randn(4, 1, 256, 256))
        target = torch.randint(0, 2, (4, 1, 256, 256)).float()
        tracker.update(pred, target)

    tracker.print_metrics()
