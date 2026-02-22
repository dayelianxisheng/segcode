import os, argparse, torch
import time

from torch import nn, optim
from torch.utils.data import DataLoader
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model import UnetPP
from model import DiceLoss, CombinedLoss, DeepSupervisionLoss
from data import AirbusDataset
from utils import SegmentationMetrics, EarlyStopping, TrainingLogger, count_parameters
from tools.common import get_device, set_seed, create_optimizer, create_scheduler, save_checkpoint, load_checkpoint

def parse_args():
    p = argparse.ArgumentParser(description="UNet Training")
    p.add_argument("--data_path", type=str, default=r"F:\resource\data\airbusship\AirbusShip_filtered")
    p.add_argument("--weight_path", type=str, default="params/unet_ship.pth")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--result_path", type=str, default="result")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--use_augmentation", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--lr_scheduler_patience", type=int, default=5)
    p.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print_interval", type=int, default=10)
    return p.parse_args()

def train_one_epoch(model,loader,criterion,optimizer,device,interval):
    model.train()
    total=0
    for i,(x,y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total+=loss.item()
        if(i+1) %interval==0:
            print(f"  Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}")
    return total/len(loader)

def validate(model,loader,criterion,device,metrics):
    model.eval()
    total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total+=loss.item()
            # 如果pred是列表(深度监督)，取最后一个输出用于计算指标
            pred_for_metrics = pred[-1] if isinstance(pred, list) else pred
            metrics.update(pred_for_metrics, y)

    return total/len(loader),metrics.get_metrics()

def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.weight_path) or ".", exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    logger = TrainingLogger(log_dir=args.log_dir, log_to_file=True)

    model = UnetPP(in_ch=3, out_ch=1, deepsupervision=True).to(device)
    params = count_parameters(model)
    logger.model_info("UnetPP",params, device)

    criterion = DeepSupervisionLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    image_size = tuple(args.image_size)
    train_ds = AirbusDataset(args.data_path, "training", image_size, args.use_augmentation)
    val_ds =AirbusDataset(args.data_path, "validation", image_size, args.use_augmentation)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True)

    logger.dataset_info(len(train_ds), len(val_ds), args.batch_size, args.val_batch_size, image_size, args.num_workers)

    optimizer = create_optimizer(model,args.lr)
    scheduler = create_scheduler(optimizer,args.lr_scheduler_patience,args.lr_scheduler_factor)
    early_stop = EarlyStopping(args.early_stop_patience, 0.001, "min")

    checkpoint_path = args.resume if args.resume else args.weight_path
    start_epoch,best = load_checkpoint(checkpoint_path,model,optimizer,device)
    best_epoch =1
    old_lr = args.lr
    print(f"training start epochs :{args.epochs}")
    start =time.time()
    for epoch in range(start_epoch,args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device,args.print_interval)
        print(f"Train Loss: {train_loss:.4f}")
        metrics_tracker =SegmentationMetrics()
        val_loss,metrics = validate(model,val_loader,criterion,device,metrics_tracker)
        print(f"Val Loss: {val_loss:.4f},IOU{metrics['iou']:.4f}")

        is_best = val_loss < best
        if is_best:
            best, best_epoch = val_loss, epoch
            print(f"*** Best model saved, Val Loss: {best:.4f} ***")

        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
        old_lr = new_lr

        save_checkpoint(model, optimizer, epoch, val_loss, is_best, checkpoint_path, best_val_loss=best)

        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    total = time.time() - start
    print(f"\nTraining complete in {total / 60:.1f} minutes")
    print(f"Best epoch: {best_epoch}, Best Val Loss: {best:.4f}")


if __name__ == '__main__':
    main()