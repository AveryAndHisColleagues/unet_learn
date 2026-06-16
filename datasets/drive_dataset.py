from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class DriveDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_train = is_train

        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "1st_manual"

        self.image_paths = sorted(list(self.image_dir.glob("*.tif")))

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        number = image_path.stem.split("_")[0]
        mask_path = self.mask_dir / f"{number}_manual1.gif"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 先统一 resize，输入尺寸仍然固定为 image_size
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # 训练集：image 和 mask 同步做几何增强
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(
                mask,
                angle,
                interpolation=transforms.InterpolationMode.NEAREST
            )

            # 颜色增强只作用于 image
            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > 0.5).float()

        return image, mask


if __name__ == "__main__":
    dataset = DriveDataset(
        r"E:\CS\PostG\个人\unet_learn\DRIVE\training",
        image_size=256,
        is_train=True
    )

    print("dataset size:", len(dataset))

    image, mask = dataset[0]

    print("image:", image.shape)
    print("mask :", mask.shape)
    print("image min/max:", image.min().item(), image.max().item())
    print("mask unique:", torch.unique(mask))