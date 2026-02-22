"""
数据集定义
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


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



class AirbusDataset(Dataset):
    """Airbus船舶检测数据集"""
    def __init__(self, root_path, split='training', image_size=(256, 256),
                 use_augmentation=False):
        """
        Args:
            root_path: 数据集根目录
            split: 'training' 或 'validation'
            image_size: 图像尺寸
            use_augmentation: 是否使用数据增强
        """
        self.root_path = root_path
        self.split = split
        self.image_dir = os.path.join(root_path, 'images', split)
        self.mask_dir = os.path.join(root_path, 'annotations', split)
        self.image_size = image_size
        self.use_augmentation = use_augmentation

        # 获取所有图片文件名
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        print(f"加载 {split} 集: {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图片名称
        img_name = self.images[idx]
        name_id = img_name.replace('.jpg', '')

        # 构造路径
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, name_id + '.png')

        # 读取图片和mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 1. 先对image和mask都应用resize（保持相同的几何变换）
        resize = ResizeWithPadding(self.image_size)
        image = resize(image)
        mask = resize(mask)

        # 2. 应用数据增强（对image和mask应用相同的变换）
        if self.use_augmentation:
            # 随机水平翻转
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            # 随机垂直翻转
            if torch.rand(1).item() < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            # 随机旋转
            if torch.rand(1).item() < 0.5:
                angle = torch.randint(-15, 15, (1,)).item()
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)

        # 3. 转换为tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_path = r"F:\resource\data\airbusship\AirbusShip_filtered"
    dataset = AirbusDataset(data_path, split='training')
    print(f"数据集大小: {len(dataset)}")

    img, mask = dataset[0]
    print(f"图像类型: {type(img)}, 形状: {img.shape if hasattr(img, 'shape') else 'N/A'}")
    print(f"掩码类型: {type(mask)}, 形状: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
