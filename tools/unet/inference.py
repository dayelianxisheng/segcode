import os, argparse, torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UNet
from tools.common import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str, required=True)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="result")
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def resize(img, size):
    """图像等比例缩放和填充"""
    w, h = img.size
    scale = min(size[0]/w, size[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("RGB", size, (0, 0, 0)) if img.mode != "L" else Image.new("L", size, 0)
    ox, oy = (size[0]-nw)//2, (size[1]-nh)//2
    new_img.paste(img_r, (ox, oy))
    return new_img


def infer_one(model, img_path, out_dir, size, threshold, device):
    print(f"Processing: {img_path}")
    # 1. 加载并转换图片为 RGB
    img = Image.open(img_path).convert("RGB")
    orig = img.size  # 记录原始尺寸，以便最后还原

    # 2. 预处理：缩放填充，并转换为张量（Tensor），增加 Batch 维度 [1, 3, H, W]
    img_r = resize(img, size)
    x = transforms.ToTensor()(img_r).unsqueeze(0).to(device)

    # 3. 执行推理
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算，节省显存和计算资源
        pred = model(x)

    # 4. 后处理
    pred = pred.squeeze().cpu()  # 移除多余的 Batch 维度并转回 CPU
    # 阈值处理：将 0~1 之间的概率值转换为 0 或 1 (二值化)
    pred_bin = (pred > threshold).float()

    # 5. 可视化对比图生成
    x_t = transforms.ToTensor()(img_r)  # 原始缩放图
    # 将灰度的预测图和二值图重复3次，模拟 RGB 通道以便拼接
    pred_expanded = pred.unsqueeze(0).repeat(3, 1, 1)
    pred_bin_expanded = pred_bin.unsqueeze(0).repeat(3, 1, 1)

    # 堆叠在一起：[原图, 概率图, 二值图] 生成对比长图
    comp = torch.stack([x_t, pred_expanded, pred_bin_expanded], 0)
    name = os.path.basename(img_path).replace(".jpg", "_comparison.png")
    save_image(comp, os.path.join(out_dir, name))

    # 6. 保存最终掩码（还原到原始图像尺寸）
    mask = transforms.ToPILImage()(pred)
    mask.resize(orig, Image.BILINEAR)
    mask_name = os.path.basename(img_path).replace(".jpg", "_mask.png")
    mask.save(os.path.join(out_dir, mask_name))


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = UNet().to(device)

    # 加载训练好的模型权重
    if os.path.exists(args.weight):
        ckpt = torch.load(args.weight, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"Loaded: {args.weight}")

    # 判断推理模式：单张图片或是批量处理文件夹
    if args.image:
        infer_one(model, args.image, args.output_dir, tuple(args.image_size), args.threshold, device)
    else:
        d = args.image_dir or r"F:\resource\data\airbusship\AirbusShip_filtered\images\validation"
        imgs = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))][:5]
        for img_name in imgs:
            infer_one(model, os.path.join(d, img_name), args.output_dir, tuple(args.image_size), args.threshold, device)
    print(f"Done! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
