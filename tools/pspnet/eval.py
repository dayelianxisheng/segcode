"""PSPNet Evaluation Script"""
import os, argparse, torch
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import PSPNet, PSPNetWithAux
from data import AirbusDataset
from utils import SegmentationMetrics
from tools.common import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="pspnet", choices=["pspnet", "pspnet_aux"])
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    p.add_argument("--data_path", type=str, default=r"F:esource\data\airbusship\AirbusShip_filtered")
    p.add_argument("--weight", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    
    val_ds = AirbusDataset(args.data_path, "validation", tuple(args.image_size), False)
    loader = DataLoader(val_ds, args.batch_size, False, args.num_workers, True)
    
    if args.model == "pspnet":
        from model import PSPNet
        model = PSPNet(num_classes=1, backbone=args.backbone, pretrained=False).to(device)
        use_aux = False
    else:
        from model import PSPNetWithAux
        model = PSPNetWithAux(num_classes=1, backbone=args.backbone, pretrained=False).to(device)
        use_aux = True
    
    if os.path.exists(args.weight):
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        print(f"Loaded: {args.weight}")
    
    metrics_tracker = SegmentationMetrics()
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_aux:
                out, _ = model(x)
            else:
                out = model(x)
            metrics_tracker.update(out, y, args.threshold)
    
    m = metrics_tracker.get_metrics()
    print(f"IoU: {m['iou']:.4f}, Dice: {m['dice']:.4f}")
    print(f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")


if __name__ == "__main__":
    main()
