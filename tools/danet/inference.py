"""
DANet推理脚本 - 增强版
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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import DANet, DANetSimple
from tools.common import get_device


def parse_args():
    p = argparse.ArgumentParser(description="DANet Inference")
    p.add_argument("--model", type=str, default="danet_simple", choices=["danet", "danet_simple"])
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    p.add_argument("--weight", type=str, required=True)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="result")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_all", action="store_true")
    return p.parse_args()


def resize(img, size):
    """保持宽高比的resize操作"""
    w, h = img.size
    scale = min(size[0]/w, size[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("RGB", size, (0, 0, 0))
    ox, oy = (size[0]-nw)//2, (size[1]-nh)//2
    new_img.paste(img_r, (ox, oy))
    return new_img


def create_colormap_probability(prob_map):
    """创建概率图的热力图"""
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    return cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)


def create_overlay(image, mask, alpha=0.4):
    """创建叠加图"""
    overlay = image.copy()
    color_mask = np.zeros_like(overlay)
    color_mask[:, :, 2] = mask  # 红色通道
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    return overlay


def create_comprehensive_visualization(original_img, prob_map, binary_mask, overlay_img,
                                        threshold=0.5, title="DANet"):
    """创建综合可视化图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{title} Segmentation Results', fontsize=16, fontweight='bold')

    # 原图
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 概率热力图
    prob_heatmap = create_colormap_probability(prob_map)
    axes[0, 1].imshow(cv2.cvtColor(prob_heatmap, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Probability Map (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 二值掩码
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title(f'Binary Mask (threshold={threshold})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 叠加图
    axes[1, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay (Red = Prediction)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


def save_single_images(original_img, prob_map, binary_mask, overlay_img, output_dir, base_name):
    """保存单独的图像"""
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), original_img)

    # 灰度概率图
    prob_gray = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_probability.png"), prob_gray)

    # 概率热力图
    prob_heatmap = create_colormap_probability(prob_map)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_probability_heatmap.png"), prob_heatmap)

    # 二值掩码
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), binary_mask)

    # 叠加图
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay_img)


def infer_one(model, img_path, out_dir, size, threshold, device, use_multi_output, save_all=False):
    """对单张图像进行推理"""
    print(f"处理图像: {img_path}")
    base_name = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")

    # 读取图像
    img = Image.open(img_path).convert("RGB")
    original_img = cv2.imread(img_path)
    orig_size = img.size

    # 预处理
    img_r = resize(img, size)
    from torchvision import transforms
    x = transforms.ToTensor()(img_r).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        if use_multi_output:
            pred, pa_out, ca_out = model(x)
        else:
            pred = model(x)

    # 后处理
    prob_map = pred.squeeze().cpu().numpy()
    prob_map_resized = cv2.resize(prob_map, (original_img.shape[1], original_img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map_resized > threshold).astype(np.uint8) * 255
    overlay_img = create_overlay(original_img, binary_mask)

    # 保存综合可视化图
    model_name = "DANet" if use_multi_output else "DANetSimple"
    fig = create_comprehensive_visualization(original_img, prob_map_resized, binary_mask,
                                              overlay_img, threshold, model_name)
    fig.savefig(os.path.join(out_dir, f"{base_name}_comprehensive.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 保存单独的图像
    if save_all:
        save_single_images(original_img, prob_map_resized, binary_mask, overlay_img, out_dir, base_name)

    # 统计信息
    fg_pixels = np.sum(binary_mask > 0)
    total = binary_mask.size
    print(f"  前景: {fg_pixels:,}/{total:,} ({fg_pixels/total*100:.2f}%)")
    print(f"  → {base_name}_comprehensive.png")


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建模型
    if args.model == "danet":
        model = DANet(num_classes=1, backbone=args.backbone, pretrained=False).to(device)
        use_multi_output = True
        model_name = "DANet"
    else:
        model = DANetSimple(num_classes=1, backbone=args.backbone, pretrained=False).to(device)
        use_multi_output = False
        model_name = "DANetSimple"

    print(f"使用模型: {model_name}")
    print(f"Backbone: {args.backbone}")

    # 加载权重
    if os.path.exists(args.weight):
        ckpt = torch.load(args.weight, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"✓ 模型权重已加载: {args.weight}")

    # 推理
    if args.image:
        # 单张图像
        infer_one(model, args.image, args.output_dir, tuple(args.image_size),
                  args.threshold, device, use_multi_output, args.save_all)
    else:
        # 批量推理
        d = args.image_dir or r"F:\resource\data\airbusship\AirbusShip_filtered\images\validation"
        imgs = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]
        print(f"找到 {len(imgs)} 张图像")
        for img_name in imgs:
            infer_one(model, os.path.join(d, img_name), args.output_dir,
                      tuple(args.image_size), args.threshold, device, use_multi_output, args.save_all)

    print(f"\n完成! 结果保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
