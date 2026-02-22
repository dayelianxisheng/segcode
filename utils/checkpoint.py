"""
模型检查点管理
支持保存最佳模型、早停、断点续训
"""
import os
import torch
import shutil


def save_checkpoint(model, optimizer, epoch, loss, is_best=False, filepath='checkpoint.pth', best_val_loss=None):
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前loss
        is_best: 是否是最佳模型
        filepath: 保存路径
        best_val_loss: 最佳验证loss
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss

    torch.save(checkpoint, filepath)

    if is_best:
        # 保存最佳模型
        best_filepath = filepath.replace('.pth', '_best.pth')
        shutil.copy(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载检查点

    Args:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        start_epoch: 开始epoch
    """
    if not os.path.exists(filepath):
        print(f"⚠️  检查点不存在: {filepath}")
        return 0

    print(f"📂 加载检查点: {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint.get('loss', 0)

    print(f"   Epoch: {checkpoint['epoch']}, Loss: {loss:.4f}")
    print(f"   将从epoch {start_epoch}继续训练")

    return start_epoch


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: 容忍epoch数
            min_delta: 最小改善幅度
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        """
        Args:
            current_score: 当前验证指标

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
        else:
            if self.mode == 'min':
                improved = current_score < self.best_score - self.min_delta
            else:
                improved = current_score > self.best_score + self.min_delta

            if improved:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"  早停触发: {self.patience}个epoch无改善")

        return self.early_stop


if __name__ == '__main__':
    # 测试早停
    print("=" * 60)
    print("测试早停机制")
    print("=" * 60)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    scores = [0.8, 0.75, 0.72, 0.71, 0.70, 0.69, 0.68]

    for epoch, score in enumerate(scores, 1):
        print(f"\nEpoch {epoch}: Score = {score:.4f}")
        if early_stopping(score):
            print(f" 停止训练（在epoch {epoch}）")
            break
        else:
            print(f"   继续训练（counter: {early_stopping.counter}/{early_stopping.patience}）")

    print(f"\n最佳分数: {early_stopping.best_score:.4f}")
