"""
UNet3Plus Inference Script
对单张图像或目录中的图像进行推理
"""
import os
import argparse
import torch
import sys
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UNet3Plus
from tools.common import get_device


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="UNet3Plus Inference")
    p.add_argument("--input", type=str, required=True,
                   help="输入图像路径或目录")
    p.add_argument("--output", type=str, default="result/inference",
                   help="输出目录路径")
    p.add_argument("--weight_path", type=str, default="params/unet3p_ship_best.pth",
                   help="模型权重路径")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                   help="输入图像尺寸 (高 宽)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="二值化阈值")
    p.add_argument("--overlay", action="store_true",
                   help="是否生成叠加可视化结果")
    return p.parse_args()


class ResizeWithPadding:
    """保持宽高比调整图像大小，不足部分用黑色填充"""
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img):
        # 计算缩放比例
        img_w, img_h = img.size
        target_w, target_h = self.size

        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建黑色背景
        if img.mode == 'L':
            background = Image.new('L', self.size, 0)
        else:
            background = Image.new('RGB', self.size, (0, 0, 0))

        # 居中粘贴
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(img_resized, (offset_x, offset_y))

        return background


def preprocess_image(image_path, image_size):
    """
    预处理图像

    Args:
        image_path: 图像路径
        image_size: 目标尺寸

    Returns:
        tensor, 原始PIL图像
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')

    # 保存原始图像用于可视化
    original_image = image.copy()

    # 调整大小
    resize = ResizeWithPadding(image_size)
    image = resize(image)

    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, original_image


def postprocess_prediction(pred_tensor, original_image, threshold=0.5):
    """
    后处理预测结果

    Args:
        pred_tensor: 预测tensor (1, 1, H, W)
        original_image: 原始图像
        threshold: 二值化阈值

    Returns:
        掩码PIL图像, 叠加图像(如果需要)
    """
    # 二值化
    pred_mask = (pred_tensor > threshold).float()

    # 转换为numpy并保存为图像
    mask_np = pred_mask.squeeze().cpu().numpy()
    mask_image = Image.fromarray((mask_np * 255).astype('uint8'), mode='L')

    return mask_image


def create_overlay(original_image, mask_image, alpha=0.5):
    """
    创建叠加可视化

    Args:
        original_image: 原始图像
        mask_image: 掩码图像
        alpha: 透明度

    Returns:
        叠加图像
    """
    # 调整mask到原始图像大小
    mask_resized = mask_image.resize(original_image.size, Image.Resampling.NEAREST)

    # 将mask转为RGB (红色表示前景)
    mask_rgb = Image.new('RGB', mask_resized.size)
    for y in range(mask_resized.size[1]):
        for x in range(mask_resized.size[0]):
            if mask_resized.getpixel((x, y)) > 128:
                mask_rgb.putpixel((x, y), (255, 0, 0))

    # 叠加
    overlay = Image.blend(original_image, mask_rgb, alpha)
    return overlay


def infer_single_image(model, image_path, output_dir, args):
    """
    对单张图像进行推理

    Args:
        model: 模型
        image_path: 图像路径
        output_dir: 输出目录
        args: 参数
    """
    # 预处理
    image_tensor, original_image = preprocess_image(image_path, tuple(args.image_size))
    image_tensor = image_tensor.to(next(model.parameters()).device)

    # 推理
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        if isinstance(pred, list):
            pred = pred[0]  # 如果是深度监督输出，取第一个

    # 后处理
    mask_image = postprocess_prediction(pred, original_image, args.threshold)

    # 保存结果
    image_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # 保存掩码
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    mask_image.save(mask_path)

    # 如果需要叠加可视化
    if args.overlay:
        overlay_image = create_overlay(original_image, mask_image)
        overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")
        overlay_image.save(overlay_path)
        print(f"已保存: {mask_path}, {overlay_path}")
    else:
        print(f"已保存: {mask_path}")


def main():
    """主推理函数"""
    args = parse_args()

    # 设备配置
    device = get_device()

    print("=" * 60)
    print("UNet3Plus 模型推理")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"权重路径: {args.weight_path}")
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"阈值: {args.threshold}")
    print("=" * 60)

    # 创建模型
    model = UNet3Plus(in_channels=3, num_classes=1).to(device)

    # 加载权重
    if not os.path.exists(args.weight_path):
        print(f"错误: 权重文件不存在: {args.weight_path}")
        return

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载权重 (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 处理输入
    input_path = Path(args.input)

    if input_path.is_file():
        # 单张图像
        print(f"\n处理图像: {args.input}")
        infer_single_image(model, args.input, args.output, args)

    elif input_path.is_dir():
        # 目录中的所有图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir()
                       if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"错误: 目录中没有找到图像文件: {args.input}")
            return

        print(f"\n找到 {len(image_files)} 张图像")
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {image_file.name}")
            infer_single_image(model, str(image_file), args.output, args)

    else:
        print(f"错误: 输入路径不存在: {args.input}")

    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
