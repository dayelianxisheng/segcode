import os
import argparse
import torch
import torch.onnx
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import create_transunet


def export_to_onnx(model, onnx_path, img_size=256, batch_size=1, opset_version=14):
    """导出模型为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

    # 验证ONNX模型
    import onnx
    from onnx import checker
    model_onnx = onnx.load(onnx_path)
    checker.check_model(model_onnx)

    print(f"✓ ONNX模型已保存: {onnx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default=r"D:\resource\code\python\ai\segcode\tools\transunet\params\transunet_ship_best.pth")
    parser.add_argument("--onnx_path", type=str, default="transunet_ship.onnx")
    parser.add_argument("--variant", type=str, default="vit_b32")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    # 创建模型
    model = create_transunet(
        variant=args.variant,
        img_size=args.img_size,
        in_channels=3,
        num_classes=1
    )

    # 加载权重
    if os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    # 导出
    os.makedirs(os.path.dirname(args.onnx_path) or ".", exist_ok=True)
    export_to_onnx(model, args.onnx_path, args.img_size)


if __name__ == "__main__":
    main()
