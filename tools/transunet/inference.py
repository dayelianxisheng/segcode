"""
TransUNet推理脚本 - 增强版
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
matplotlib.use('Agg')  # 非交互式后端
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import create_transunet
from tools.common import get_device


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="TransUNet Inference")
    p.add_argument("--weight", type=str, default="params/transunet_ship_best.pth",
                   help="模型权重路径")
    p.add_argument("--image", type=str, required=True,
                   help="输入图像路径")
    p.add_argument("--output_dir", type=str, default=None,
                   help="输出目录（默认tools/transunet/result）")
    p.add_argument("--variant", type=str, default="vit_b16",
                   choices=["vit_b16", "vit_b32", "vit_l16"],
                   help="TransUNet变体")
    p.add_argument("--img_size", type=int, default=256,
                   help="输入图像尺寸")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="二值化阈值")
    p.add_argument("--save_all", action="store_true",
                   help="是否保存所有类型的图像（原图、概率图、掩码图、叠加图、综合图）")
    return p.parse_args()


def preprocess_image(image_path, img_size):
    """预处理图像"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)

    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, original_size, image


def create_colormap_probability(prob_map):
    """创建概率热力图（使用jet颜色映射）"""
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    return cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)


def create_overlay(image, mask, alpha=0.4):
    """创建叠加图"""
    overlay = image.copy()
    # 创建彩色掩码（红色）
    color_mask = np.zeros_like(overlay)
    color_mask[:, :, 2] = mask  # 红色通道

    # 混合
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    return overlay


def create_comprehensive_visualization(original_img, prob_map, binary_mask, overlay_img):
    """
    创建综合可视化图

    包含：
    - 原图
    - 概率热力图
    - 二值掩码
    - 叠加图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('TransUNet Segmentation Results', fontsize=16, fontweight='bold')

    # 1. 原图
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. 概率热力图
    prob_heatmap = create_colormap_probability(prob_map)
    axes[0, 1].imshow(cv2.cvtColor(prob_heatmap, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Probability Map (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 添加颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label('Probability', rotation=270, labelpad=15)

    # 3. 二值掩码
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title(f'Binary Mask (threshold={0.5})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 4. 叠加图
    axes[1, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay (Red = Prediction)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


def save_single_images(original_img, prob_map, binary_mask, overlay_img, output_dir, base_name):
    """保存单独的图像"""
    # 1. 原图
    orig_path = os.path.join(output_dir, f"{base_name}_original.png")
    cv2.imwrite(orig_path, original_img)

    # 2. 概率图（灰度 + 热力图）
    prob_gray_path = os.path.join(output_dir, f"{base_name}_probability.png")
    prob_gray = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(prob_gray_path, prob_gray)

    prob_heatmap_path = os.path.join(output_dir, f"{base_name}_probability_heatmap.png")
    prob_heatmap = create_colormap_probability(prob_map)
    cv2.imwrite(prob_heatmap_path, prob_heatmap)

    # 3. 二值掩码
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, binary_mask)

    # 4. 叠加图
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay_img)


def main():
    """主函数"""
    args = parse_args()

    # 设备
    device = get_device()

    # 创建输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建模型
    print(f"加载TransUNet模型: {args.variant}")
    model = create_transunet(
        variant=args.variant,
        img_size=args.img_size,
        in_channels=3,
        num_classes=1
    ).to(device)

    # 加载权重
    if os.path.exists(args.weight):
        checkpoint = torch.load(args.weight, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ 模型权重已加载: {args.weight}")
    else:
        print(f"⚠ 警告: 权重文件不存在: {args.weight}")
        print("使用未训练的模型进行推理...")

    model.eval()

    # 预处理
    print(f"\n处理图像: {args.image}")
    image_tensor, original_size, original_pil = preprocess_image(args.image, args.img_size)
    image_tensor = image_tensor.to(device)

    # 读取原始图像（用于可视化）
    original_img = cv2.imread(args.image)

    # 推理
    print("执行推理...")
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = output.squeeze().cpu().numpy()  # 概率图 (H, W)

    # 后处理
    # 1. 将概率图调整到原始尺寸
    prob_map_resized = cv2.resize(prob_map, (original_img.shape[1], original_img.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    # 2. 二值化
    binary_mask = (prob_map_resized > args.threshold).astype(np.uint8) * 255

    # 3. 创建叠加图
    overlay_img = create_overlay(original_img, binary_mask, alpha=0.4)

    # 生成文件名
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    # 保存综合可视化图
    print("\n保存结果...")
    comprehensive_path = os.path.join(args.output_dir, f"{base_name}_comprehensive.png")
    fig = create_comprehensive_visualization(original_img, prob_map_resized, binary_mask, overlay_img)
    fig.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 综合可视化图: {comprehensive_path}")

    # 保存单独的图像
    if args.save_all:
        save_single_images(original_img, prob_map_resized, binary_mask, overlay_img,
                          args.output_dir, base_name)
        print(f"✓ 单独图像已保存:")
        print(f"  - 原图: {base_name}_original.png")
        print(f"  - 概率图: {base_name}_probability.png")
        print(f"  - 概率热力图: {base_name}_probability_heatmap.png")
        print(f"  - 二值掩码: {base_name}_mask.png")
        print(f"  - 叠加图: {base_name}_overlay.png")

    # 统计信息
    foreground_pixels = np.sum(binary_mask > 0)
    total_pixels = binary_mask.size
    foreground_ratio = foreground_pixels / total_pixels * 100

    # 平均概率
    avg_prob = prob_map_resized[binary_mask > 0].mean() if foreground_pixels > 0 else 0

    print(f"\n统计信息:")
    print(f"  前景像素: {foreground_pixels:,} / {total_pixels:,} ({foreground_ratio:.2f}%)")
    print(f"  平均概率: {avg_prob:.4f}")
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
