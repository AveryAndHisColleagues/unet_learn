from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DriveDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = Path(root_dir)

        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "1st_manual"

        self.image_files = sorted(list(self.image_dir.glob("*.tif")))

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        number = image_path.stem.split("_")[0]
        mask_path = self.mask_dir / f"{number}_manual1.gif"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)

        mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    dataset = DriveDataset(
        r"E:\CS\PostG\个人\unet_learn\DRIVE\training",
        image_size=256
    )

    print("dataset size:", len(dataset))

    image, mask = dataset[0]

    print("image:", image.shape)
    print("mask :", mask.shape)
    print("image min/max:", image.min().item(), image.max().item())
    print("mask unique:", torch.unique(mask))