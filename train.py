import torch
import csv
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from datasets.drive_dataset import DriveDataset
from utils.losses import BCETverskyLoss
from utils.metrics import binary_iou

# models
from models.resunet import ResUNet
from models.resunet_lightssm import ResUNetLightSSM


# ======================
# model factory
# ======================
def build_model(name):
    if name == "resunet":
        return ResUNet(num_classes=1)

    elif name == "lightssm_v1":
        return ResUNetLightSSM(num_classes=1)

    elif name == "lightssm_v2":
        return ResUNetLightSSM(num_classes=1)

    else:
        raise ValueError("Unknown model")


# ======================
# train
# ======================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += binary_iou(preds, masks)

    return total_loss / len(loader), total_iou / len(loader)


# ======================
# validate
# ======================
@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        total_loss += loss.item()
        total_iou += binary_iou(preds, masks)

    return total_loss / len(loader), total_iou / len(loader)


# ======================
# main
# ======================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resunet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== experiment name =====
    EXP_NAME = args.model

    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    log_path = Path(f"logs/{EXP_NAME}.csv")
    best_model_path = Path(f"checkpoints/best_{EXP_NAME}.pth")

    # ===== dataset =====
    dataset = DriveDataset(
        root_dir=r"E:\CS\PostG\个人\unet_learn\DRIVE\training",
        image_size=256,
        is_train=True
    )

    n = len(dataset)
    indices = torch.randperm(n).tolist()

    train_size = int(0.8 * n)

    train_set = Subset(dataset, indices[:train_size])
    val_set = Subset(dataset, indices[train_size:])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # ===== model =====
    model = build_model(args.model).to(device)

    # ===== loss =====
    loss_fn = BCETverskyLoss(
        bce_weight=0.3,
        tversky_weight=0.7
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ===== log =====
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou", "val_loss", "val_iou", "best_iou"])

        best_iou = 0.0

        for epoch in range(args.epochs):

            train_loss, train_iou = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )

            val_loss, val_iou = validate(
                model, val_loader, loss_fn, device
            )

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), best_model_path)
                flag = "✔"
            else:
                flag = ""

            writer.writerow([epoch, train_loss, train_iou, val_loss, val_iou, best_iou])

            print(
                f"[{args.model}] E{epoch+1} "
                f"train_loss={train_loss:.4f} "
                f"train_iou={train_iou:.4f} "
                f"val_iou={val_iou:.4f} "
                f"best={best_iou:.4f} {flag}"
            )

    print("done:", EXP_NAME)
    print("best_iou:", best_iou)


if __name__ == "__main__":
    main()