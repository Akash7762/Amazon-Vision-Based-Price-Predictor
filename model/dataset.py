"""
PyTorch Dataset / DataLoader for the price-regression task (Phase 2).

Works identically whether pointed at the ~15k-row dev subset (local) or the
full ~51k-row train split (Kaggle) — nothing here is hardcoded to a row
count or a specific path. The only assumption is that images have already
been downloaded and resized by scripts/download_images.py into
`image_dir/{sample_id}.jpg`.

Rows whose image file is missing (e.g. the download failed for that
sample_id) are dropped at Dataset-construction time, with the count printed
explicitly — never silently trained on a placeholder/black image.
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization stats — matches the pretrained convnext_tiny
# backbone's expected input distribution (confirmed via
# timm.get_pretrained_cfg('convnext_tiny.fb_in22k_ft_in1k')).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class PriceImageDataset(Dataset):
    """Returns (image_tensor, price_float_tensor) pairs.

    Args:
        metadata_path: path to a metadata CSV with columns
            sample_id, image_link, price, split (dev_subset.csv or metadata.csv).
        image_dir: directory containing '{sample_id}.jpg' files, produced by
            scripts/download_images.py.
        split: which `split` column value to filter to ("train" or "val").
        image_size: square resolution images were resized to (must match
            what download_images.py used).
        train: whether to apply training-time augmentation (True) or the
            deterministic val/test transform (False).
    """

    def __init__(self, metadata_path: str, image_dir: str, split: str,
                 image_size: int = 224, train: bool = None):
        if train is None:
            train = (split == "train")

        df = pd.read_csv(metadata_path)
        df = df[df["split"] == split].reset_index(drop=True)

        # Keep only rows whose image actually downloaded successfully.
        image_paths = df["sample_id"].apply(lambda sid: os.path.join(image_dir, f"{sid}.jpg"))
        exists_mask = image_paths.apply(os.path.exists)

        n_total = len(df)
        n_missing = int((~exists_mask).sum())
        df = df[exists_mask].reset_index(drop=True)

        print(f"[PriceImageDataset] split='{split}': {n_total} rows in metadata, "
              f"{n_missing} missing local image(s) dropped, {len(df)} usable rows.")

        self.df = df
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = build_transforms(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        price = torch.tensor(row["price"], dtype=torch.float32)
        return image, price


def build_dataloaders(metadata_path: str, image_dir: str, image_size: int,
                       batch_size: int, num_workers: int = 0):
    """Build train/val DataLoaders. subset_mode is implicit in whichever
    metadata_path/image_dir is passed in (dev_subset.csv + dev image dir,
    or metadata.csv + full image dir) — this function itself has no
    subset-specific logic.
    """
    train_ds = PriceImageDataset(metadata_path, image_dir, split="train",
                                  image_size=image_size, train=True)
    val_ds = PriceImageDataset(metadata_path, image_dir, split="val",
                                image_size=image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader
