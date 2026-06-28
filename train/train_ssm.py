import torch
import csv
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from datasets.drive_dataset import DriveDataset
from models.resunet import ResUNet
from utils.losses import BCETverskyLoss
from utils.metrics import binary_iou
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== log =====
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    log_path = Path("logs/e9_ssm_log.csv")
    best_model_path = Path("checkpoints/best_ssm.pth")

    # ===== dataset =====
    dataset = DriveDataset(
        root_dir=r"E:\CS\PostG\个人\unet_learn\DRIVE\training",
        image_size=256,
        is_train=True
    )

    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()

    train_size = int(0.8 * n)

    train_set = Subset(dataset, indices[:train_size])
    val_set = Subset(dataset, indices[train_size:])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    # ===== model =====
    model = ResUNet(num_classes=1).to(device)

    # 已经在 ResUNet 里插了 SSMBottleneck
    # 所以这里不用改模型结构

    # ===== loss =====
    loss_fn = BCETverskyLoss(
        bce_weight=0.3,
        tversky_weight=0.7
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== log =====
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou", "val_loss", "val_iou", "best_iou"])

        best_iou = 0.0
        epochs = 50

        for epoch in range(epochs):

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
                f"[E{epoch+1}] "
                f"train_loss={train_loss:.4f} "
                f"train_iou={train_iou:.4f} "
                f"val_iou={val_iou:.4f} "
                f"best={best_iou:.4f} {flag}"
            )

    print("done")
    print("best_iou:", best_iou)


if __name__ == "__main__":
    main()