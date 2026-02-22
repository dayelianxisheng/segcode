"""
损失函数定义
支持BCE、Dice、Focal及组合损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失

    Dice = 2*|X∩Y| / (|X| + |Y|)
    Loss = 1 - Dice

    适用于处理类别不平衡问题
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W)，范围[0, 1]
            target: 真实标签 (B, 1, H, W)，范围[0, 1]
        """
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 计算交集
        intersection = (pred_flat * target_flat).sum()

        # Dice系数
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: 平衡因子，用于平衡正负样本
        gamma: 聚焦参数，gamma越大越关注难分类样本
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W)，范围[0, 1]
            target: 真实标签 (B, 1, H, W)，范围[0, 1]
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # 计算pt
        pt = target * pred + (1 - target) * (1 - pred)

        # Focal loss
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = at * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数

    结合BCE Loss和Dice Loss的优点
    BCE提供稳定的梯度，Dice优化分割指标
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return loss


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失函数

    用于处理UNet++等返回多尺度输出的模型
    对每个输出计算损失并加权组合
    """
    def __init__(self, weights=None, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            weights: 各输出的权重列表，如[0.1, 0.2, 0.3, 0.4]，None则使用等权重
            bce_weight: BCE损失权重
            dice_weight: Dice损失权重
            smooth: Dice损失平滑项
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights if weights is not None else [0.25, 0.25, 0.25, 0.25]
        self.combined_loss = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)

    def forward(self, preds, target):
        """
        Args:
            preds: 预测值列表 [pred1, pred2, pred3, pred4]，每个形状为 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)

        Returns:
            total_loss: 加权组合的总损失
        """
        if not isinstance(preds, list):
            return self.combined_loss(preds, target)

        total_loss = 0
        for i, pred in enumerate(preds):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            loss = self.combined_loss(pred, target)
            total_loss += weight * loss

        return total_loss


class PSPNetLoss(nn.Module):
    """
    PSPNet带辅助损失的损失函数

    结合主分支损失和辅助分支损失
    主分支使用组合损失，辅助分支使用BCE损失
    """
    def __init__(self, main_loss=None, aux_weight=0.4, use_combined=True,
                 bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        """
        Args:
            main_loss: 主分支损失函数，如果为None则使用CombinedLoss
            aux_weight: 辅助损失权重
            use_combined: 是否使用组合损失（BCE+Dice），否则只用BCE
            bce_weight: BCE损失权重（当use_combined=True时）
            dice_weight: Dice损失权重（当use_combined=True时）
            smooth: Dice损失平滑项
        """
        super(PSPNetLoss, self).__init__()
        self.aux_weight = aux_weight

        if main_loss is None:
            if use_combined:
                self.main_loss_fn = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)
            else:
                self.main_loss_fn = nn.BCELoss()
        else:
            self.main_loss_fn = main_loss

        # 辅助损失使用BCE
        self.aux_loss_fn = nn.BCELoss()

    def forward(self, pred, target, aux_pred=None):
        """
        Args:
            pred: 主分支预测 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
            aux_pred: 辅助分支预测 (B, 1, H, W)，可选

        Returns:
            loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 主分支损失
        main_loss = self.main_loss_fn(pred, target)

        # 辅助分支损失
        if aux_pred is not None:
            aux_loss = self.aux_loss_fn(aux_pred, target)
            total_loss = main_loss + self.aux_weight * aux_loss
            loss_dict = {
                'total': total_loss.item(),
                'main': main_loss.item(),
                'aux': aux_loss.item()
            }
        else:
            total_loss = main_loss
            loss_dict = {
                'total': total_loss.item(),
                'main': main_loss.item(),
                'aux': 0.0
            }

        return total_loss, loss_dict

class UNet3PlusLoss(nn.Module):
    def __init__(self, num_classes=1, smooth=1e-5, boundary_weight=0.1):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight

    def dice_loss(self, pred, target):
        """Dice损失，缓解类别不平衡"""
        if self.num_classes == 1:
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        else:
            pred = F.softmax(pred, dim=1)
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, preds, targets):
        """
        Args:
            preds: 训练时为[sup1, sup2, sup3, sup4, sup5]，推理时为sup1
            targets: 分割标签（与原图同尺度，通道数=num_classes）
        """
        if isinstance(preds, list):
            # 深度监督损失：浅层权重更高（匹配论文设计）
            loss = 0.
            # sup1 (D1)：权重1.0
            loss += self.dice_loss(preds[0], targets) + F.binary_cross_entropy_with_logits(preds[0], targets)
            # sup2 (D2)：权重0.8
            loss += 0.8 * (self.dice_loss(preds[1], targets) + F.binary_cross_entropy_with_logits(preds[1], targets))
            # sup3 (D3)：权重0.6
            loss += 0.6 * (self.dice_loss(preds[2], targets) + F.binary_cross_entropy_with_logits(preds[2], targets))
            # sup4 (D4)：权重0.4
            loss += 0.4 * (self.dice_loss(preds[3], targets) + F.binary_cross_entropy_with_logits(preds[3], targets))
            # sup5 (D5)：权重0.2
            loss += 0.2 * (self.dice_loss(preds[4], targets) + F.binary_cross_entropy_with_logits(preds[4], targets))
            return loss / 5.  # 平均损失
        else:
            # 推理时仅计算主输出损失
            dice = self.dice_loss(preds, targets)
            ce = F.binary_cross_entropy_with_logits(preds, targets)
            return dice + ce


if __name__ == '__main__':
    # 测试损失函数
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)

    # 创建模拟数据
    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    # 测试各种损失
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    combined_loss = CombinedLoss()

    print(f"BCE Loss: {bce_loss(pred, target).item():.4f}")
    print(f"Dice Loss: {dice_loss(pred, target).item():.4f}")
    print(f"Focal Loss: {focal_loss(pred, target).item():.4f}")
    print(f"Combined Loss: {combined_loss(pred, target).item():.4f}")
