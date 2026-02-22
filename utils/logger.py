import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


class TrainingLogger:
    """训练日志记录器"""

    # ANSI颜色代码
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m',
    }

    def __init__(self, log_dir='logs', log_to_file=True, use_colors=True):
        """
        Args:
            log_dir: 日志文件保存目录
            log_to_file: 是否保存日志到文件
            use_colors: 是否使用彩色输出（Windows可能需要额外配置）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.log_to_file = log_to_file
        self.use_colors = use_colors and self._supports_color()

        # 创建日志文件
        self.log_file = None
        if log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = self.log_dir / f'train_{timestamp}.log'

        # 训练统计
        self.epoch_start_time = None
        self.train_start_time = None
        self.step_times = []
        self.total_steps = 0

    def _supports_color(self):
        """检测终端是否支持颜色"""
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                return False
        return True

    def _colorize(self, text, color):
        """给文本添加颜色"""
        if self.use_colors:
            return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
        return text

    def _write(self, message):
        """写入日志文件"""
        if self.log_to_file and self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # 移除ANSI颜色代码
                clean_message = message
                for color_code in self.COLORS.values():
                    clean_message = clean_message.replace(color_code, '')
                f.write(clean_message + '\n')

    def info(self, message, color=None):
        """输出信息日志"""
        if color:
            message = self._colorize(message, color)
        print(message)
        self._write(message)

    def header(self, title):
        """输出标题"""
        width = 70
        border = '=' * width
        self.info(border)
        self.info(f'{title:^{width}}')
        self.info(border)

    def separator(self, char='-', width=70):
        """输出分隔线"""
        self.info(char * width)

    # ========== 训练阶段 ==========

    def train_start(self, total_epochs):
        """训练开始"""
        self.train_start_time = time.time()
        self.info("")
        self.header("训练开始")
        self.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"训练轮数: {total_epochs}")
        if self.log_to_file:
            self.info(f"日志文件: {self.log_file}")
        self.separator()

    def train_end(self):
        """训练结束"""
        total_time = time.time() - self.train_start_time
        self.separator()
        self.header("训练完成")
        self.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"总训练时长: {self._format_time(total_time)}")
        self.separator()

    # ========== Epoch阶段 ==========

    def epoch_start(self, epoch, total_epochs):
        """Epoch开始"""
        self.epoch_start_time = time.time()
        self.step_times = []
        self.info("")
        self.info(f"Epoch [{epoch}/{total_epochs}]")

    def epoch_end(self, epoch, total_epochs, train_loss, val_loss, metrics,
                  learning_rate, is_best=False):
        """Epoch结束"""
        epoch_time = time.time() - self.epoch_start_time

        # 构建输出
        lines = [
            "",
            f"  时间: {self._format_time(epoch_time)}",
            f"  学习率: {learning_rate:.2e}",
            f"  训练Loss: {self._colorize(f'{train_loss:.4f}', 'cyan')}",
            f"  验证Loss: {val_loss:.4f}",
            f"  IoU: {metrics['iou']:.4f}",
            f"  Dice: {metrics['dice']:.4f}",
            f"  Precision: {metrics['precision']:.4f}",
            f"  Recall: {metrics['recall']:.4f}",
        ]

        if is_best:
            lines.append(f"  {self._colorize('★ 最佳模型!', 'green')}")

        # 进度条
        progress = self._progress_bar(epoch, total_epochs, width=50)
        lines.append(f"  进度: {progress}")

        for line in lines:
            self.info(line.rstrip())

    # ========== 训练步骤 ==========

    def step_update(self, step, total_steps, loss, learning_rate=None,
                    phase="Train", show_every=10):
        """更新训练步骤"""
        self.total_steps += 1

        if step % show_every == 0 or step == total_steps - 1:
            # 计算ETA
            if self.step_times:
                avg_step_time = sum(self.step_times) / len(self.step_times)
                remaining_steps = total_steps - step - 1
                eta = avg_step_time * remaining_steps
            else:
                eta = 0

            # 进度条
            progress = self._progress_bar(step + 1, total_steps, width=30)

            # 构建输出
            lr_str = f" LR:{learning_rate:.2e}" if learning_rate else ""
            eta_str = f" ETA:{self._format_time(eta)}" if eta > 0 else ""

            msg = (f"  [{phase}] {progress} Step:{step+1}/{total_steps} "
                   f"Loss:{loss:.4f}{lr_str}{eta_str}")

            # 使用\r覆盖（只在终端显示）
            if step == total_steps - 1:
                self.info(msg)  # 最后一步换行
            else:
                print(f"\r{msg}", end='', flush=True)
                self._write(msg)  # 日志文件每次都写新行

        self.step_times.append(time.time() - self.epoch_start_time)

    # ========== 验证阶段 ==========

    def validation_start(self):
        """验证开始"""
        self.info("  验证中...", 'gray')

    def validation_end(self, val_loss, metrics):
        """验证结束"""
        self.info(f"  验证Loss: {val_loss:.4f} | "
                 f"IoU:{metrics['iou']:.4f} "
                 f"Dice:{metrics['dice']:.4f} "
                 f"P:{metrics['precision']:.4f} "
                 f"R:{metrics['recall']:.4f}")

    # ========== 最佳模型 ==========

    def best_model(self, val_loss, epoch):
        """最佳模型更新"""
        self.info(f"  {self._colorize('✓', 'green')} 最佳验证Loss更新: "
                 f"{val_loss:.4f} (Epoch {epoch})")

    # ========== 早停 ==========

    def early_stop(self, epoch):
        """早停触发"""
        self.info("")
        self.info(f"{self._colorize('Early Stop', 'yellow')} "
                 f"在 Epoch {epoch} 触发早停")

    # ========== 学习率调度 ==========

    def lr_reduce(self, old_lr, new_lr, epoch):
        """学习率降低"""
        self.info(f"  {self._colorize('↓', 'yellow')} 学习率降低: "
                 f"{old_lr:.2e} → {new_lr:.2e} (Epoch {epoch})")

    # ========== 检查点 ==========

    def checkpoint_save(self, path, is_best=False):
        """保存检查点"""
        suffix = " (最佳)" if is_best else ""
        self.info(f"  检查点已保存: {path}{suffix}")

    def checkpoint_load(self, path, epoch, best_val_loss):
        """加载检查点"""
        self.info(f"从检查点恢复:")
        self.info(f"  路径: {path}")
        self.info(f"  Epoch: {epoch}")
        self.info(f"  最佳验证Loss: {best_val_loss:.4f}")

    # ========== 工具方法 ==========

    def _format_time(self, seconds):
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _progress_bar(self, current, total, width=40):
        """创建进度条"""
        filled = int(width * current / total)
        bar = '█' * filled + '░' * (width - filled)
        percentage = current / total * 100
        return f"[{bar}] {percentage:.0f}%"

    # ========== 数据集信息 ==========

    def dataset_info(self, train_size, val_size, batch_size, val_batch_size,
                     image_size, num_workers):
        """输出数据集信息"""
        self.info("")
        self.info("=" * 70)
        self.info("数据集配置:")
        self.info(f"  训练集: {train_size:,} 张")
        self.info(f"  验证集: {val_size:,} 张")
        self.info(f"  批次大小: {batch_size} (训练) / {val_batch_size} (验证)")
        self.info(f"  图像尺寸: {image_size[0]}×{image_size[1]}")
        self.info(f"  工作进程: {num_workers}")
        self.info("=" * 70)

    # ========== 模型信息 ==========

    def model_info(self, model_name, num_params, device):
        """输出模型信息"""
        self.info("")
        self.info("=" * 70)
        self.info("模型配置:")
        self.info(f"  模型: {model_name}")
        self.info(f"  参数量: {num_params:,}")
        self.info(f"  设备: {device}")
        self.info("=" * 70)

    # ========== 训练摘要 ==========

    def summary(self, best_epoch, best_val_loss, best_metrics, total_epochs,
                total_time):
        """输出训练摘要"""
        self.info("")
        self.header("训练摘要")
        self.info(f"  最佳 Epoch: {best_epoch}/{total_epochs}")
        self.info(f"  最佳验证Loss: {best_val_loss:.4f}")
        self.info(f"  最佳IoU: {best_metrics['iou']:.4f}")
        self.info(f"  最佳Dice: {best_metrics['dice']:.4f}")
        self.info(f"  总训练时间: {self._format_time(total_time)}")
        self.info(f"  平均每Epoch: {self._format_time(total_time / total_epochs)}")
        self.separator()


class ProgressBar:
    """简单的进度条"""

    def __init__(self, total, description='', width=40):
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        """更新进度"""
        self.current += n
        self._display()

    def _display(self):
        """显示进度条"""
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        percentage = self.current / self.total * 100

        # 计算ETA
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0

        print(f"\r{self.description} [{bar}] {percentage:.1f}% "
              f"ETA:{TrainingLogger._format_time(None, eta)}",
              end='', flush=True)

    @staticmethod
    def _format_time(seconds):
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

    def close(self):
        """关闭进度条"""
        print()  # 换行


# 便捷函数
def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试日志工具
    logger = TrainingLogger(log_to_file=False)

    logger.train_start(10)
    logger.model_info("UNet", 31000000, "cuda")
    logger.dataset_info(10000, 2000, 8, 4, (256, 256), 0)

    for epoch in range(1, 4):
        logger.epoch_start(epoch, 10)

        for step in range(50):
            logger.step_update(step, 50, 0.5 - epoch * 0.1 - step * 0.001,
                              learning_rate=0.001, show_every=10)
            time.sleep(0.01)

        logger.epoch_end(epoch, 10, 0.3, 0.25,
                        {'iou': 0.75, 'dice': 0.85, 'precision': 0.88, 'recall': 0.82},
                        0.001, is_best=(epoch == 2))

        if epoch == 2:
            logger.lr_reduce(0.001, 0.0005, epoch)

    logger.early_stop(3)
    logger.summary(2, 0.25, {'iou': 0.75, 'dice': 0.85, 'precision': 0.88, 'recall': 0.82},
                  3, 120)
    logger.train_end()
