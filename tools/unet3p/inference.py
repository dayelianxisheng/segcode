"""
UNet3Plus推理脚本 - 增强版
支持原图、概率图、掩码图、叠加图的绘制
"""
import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UNet3Plus
from tools.common import get_device


def parse_args():
    p = argparse.ArgumentParser(description="UNet3Plus Inference")
    p.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    p.add_argument("--output", type=str, default="result/inference", help="输出目录路径")
    p.add_argument("--weight_path", type=str, default="params/unet3p_ship_best.pth", help="模型权重路径")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="输入图像尺寸")
    p.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    p.add_argument("--save_all", action="store_true", help="保存所有类型的图像")
    return p.parse_args()


class ResizeWithPadding:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img):
        img_w, img_h = img.size
        target_w, target_h = self.size
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        background = Image.new('RGB', self.size, (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(img_resized, (offset_x, offset_y))
        return background


def create_colormap_probability(prob_map):
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    return cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)


def create_overlay(image, mask, alpha=0.4):
    overlay = image.copy()
    color_mask = np.zeros_like(overlay)
    color_mask[:, :, 2] = mask
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    return overlay


def create_comprehensive_visualization(original_img, prob_map, binary_mask, overlay_img, threshold=0.5):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('UNet3+ Segmentation Results', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    prob_heatmap = create_colormap_probability(prob_map)
    axes[0, 1].imshow(cv2.cvtColor(prob_heatmap, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Probability Map (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title(f'Binary Mask (threshold={threshold})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay (Red = Prediction)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


def save_single_images(original_img, prob_map, binary_mask, overlay_img, output_dir, base_name):
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), original_img)
    prob_gray = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_probability.png"), prob_gray)
    prob_heatmap = create_colormap_probability(prob_map)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_probability_heatmap.png"), prob_heatmap)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), binary_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay_img)


def infer_single_image(model, image_path, output_dir, args):
    print(f"处理图像: {image_path}")
    base_name = Path(image_path).stem

    # 预处理
    resize = ResizeWithPadding(tuple(args.image_size))
    img = Image.open(image_path).convert('RGB')
    original_img = cv2.imread(image_path)
    img_resized = resize(img)

    from torchvision import transforms
    x = transforms.ToTensor()(img_resized).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        pred = model(x)
        if isinstance(pred, list):
            pred = pred[0]

    # 后处理
    prob_map = pred.squeeze().cpu().numpy()
    prob_map_resized = cv2.resize(prob_map, (original_img.shape[1], original_img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map_resized > args.threshold).astype(np.uint8) * 255
    overlay_img = create_overlay(original_img, binary_mask)

    # 保存综合可视化
    fig = create_comprehensive_visualization(original_img, prob_map_resized, binary_mask, overlay_img, args.threshold)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{base_name}_comprehensive.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 保存单独图像
    if args.save_all:
        save_single_images(original_img, prob_map_resized, binary_mask, overlay_img, output_dir, base_name)

    # 统计
    fg_pixels = np.sum(binary_mask > 0)
    total = binary_mask.size
    print(f"  前景: {fg_pixels:,}/{total:,} ({fg_pixels/total*100:.2f}%)")
    print(f"  → {base_name}_comprehensive.png")


def main():
    args = parse_args()
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

    model = UNet3Plus(in_channels=3, num_classes=1).to(device)

    if not os.path.exists(args.weight_path):
        print(f"错误: 权重文件不存在: {args.weight_path}")
        return

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ 已加载权重 (Epoch {checkpoint.get('epoch', 'N/A')})")

    input_path = Path(args.input)

    if input_path.is_file():
        infer_single_image(model, args.input, args.output, args)
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"错误: 目录中没有找到图像文件: {args.input}")
            return

        print(f"\n找到 {len(image_files)} 张图像")
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            infer_single_image(model, str(image_file), args.output, args)
    else:
        print(f"错误: 输入路径不存在: {args.input}")

    print("\n" + "=" * 60)
    print(f"完成! 结果保存到: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
