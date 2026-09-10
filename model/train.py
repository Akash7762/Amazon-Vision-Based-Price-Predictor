"""
Configurable training script for the price-regression model (Phase 2).

Works in two modes via --subset-mode:
    dev  -> data/processed/dev_subset.csv + data/processed/images/dev
            (small local subset, CPU sanity-testing only)
    full -> data/processed/metadata.csv + data/processed/images/full
            (the real ~51k-row train split, intended for Kaggle GPU runs)

Nothing about the model, dataset, or loop differs between modes — only the
metadata/image paths and (typically) epoch count/batch size differ, all of
which are CLI arguments.

IMPORTANT: this script is designed to be run locally for a *sanity check*
only (small epoch count, small subset, CPU). Real full-scale training runs
on Kaggle, per the project's Phase 2 decisions — do not run this locally
with --subset-mode full and expect a real trained model.

Usage (local sanity check):
    python model/train.py --subset-mode dev --epochs 1 --batch-size 16 \
        --lr 1e-4 --max-batches 5 --checkpoint-dir model/checkpoints

Usage (Kaggle, full run — for reference, run there not here):
    python model/train.py --subset-mode full --epochs 15 --batch-size 64 \
        --lr 1e-4 --checkpoint-dir /kaggle/working/checkpoints
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import PriceImageDataset
from model.model import build_model, build_loss_fn, EXPECTED_INPUT_SIZE, BACKBONE_NAME


SUBSET_PATHS = {
    "dev": {
        "metadata": "data/processed/dev_subset.csv",
        "image_dir": "data/processed/images/dev",
    },
    "full": {
        "metadata": "data/processed/metadata.csv",
        "image_dir": "data/processed/images/full",
    },
}


def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


def run_epoch(model, loader, loss_fn, optimizer, device, logger, epoch, split_name,
              max_batches=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    start = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, prices) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            prices = prices.to(device).unsqueeze(1)  # (B,) -> (B,1) to match model output

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, prices)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            n_batches += 1

            logger.info(
                f"epoch={epoch} split={split_name} batch={batch_idx} "
                f"batch_loss={loss.item():.4f}"
            )

    elapsed = time.time() - start
    avg_loss = total_loss / max(n_samples, 1)
    logger.info(
        f"epoch={epoch} split={split_name} DONE avg_loss={avg_loss:.4f} "
        f"batches={n_batches} samples={n_samples} elapsed_sec={elapsed:.1f}"
    )
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train the price-regression model.")
    parser.add_argument("--subset-mode", choices=["dev", "full"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="model/checkpoints")
    parser.add_argument("--log-path", default="model/logs/train.log")
    parser.add_argument("--max-batches", type=int, default=None,
                         help="Cap batches per epoch (for quick sanity checks). None = full epoch.")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    paths = SUBSET_PATHS[args.subset_mode]
    logger = setup_logging(args.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Training run start ===")
    logger.info(f"subset_mode={args.subset_mode} device={device} "
                f"cuda_available={torch.cuda.is_available()}")
    logger.info(f"backbone={BACKBONE_NAME} input_size={EXPECTED_INPUT_SIZE} "
                f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
                f"max_batches={args.max_batches}")
    logger.info(f"metadata={paths['metadata']} image_dir={paths['image_dir']}")

    train_ds = PriceImageDataset(paths["metadata"], paths["image_dir"], split="train",
                                  image_size=EXPECTED_INPUT_SIZE, train=True)
    val_ds = PriceImageDataset(paths["metadata"], paths["image_dir"], split="val",
                                image_size=EXPECTED_INPUT_SIZE, train=False)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No usable training rows found for subset_mode='{args.subset_mode}'. "
            f"Did you run scripts/download_images.py for this metadata/image_dir pair?"
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"No usable validation rows found for subset_mode='{args.subset_mode}'. "
            f"Did you run scripts/download_images.py for this metadata/image_dir pair?"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(pretrained=args.pretrained).to(device)
    loss_fn = build_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                logger, epoch, "train", max_batches=args.max_batches)
        val_loss = run_epoch(model, val_loader, loss_fn, None, device,
                              logger, epoch, "val", max_batches=args.max_batches)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Save a checkpoint every epoch (so a Kaggle run can resume after an
        # interruption), plus track the best-val-loss checkpoint separately.
        epoch_ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }, epoch_ckpt_path)
        logger.info(f"Saved checkpoint: {epoch_ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args),
            }, best_ckpt_path)
            logger.info(f"New best val_loss={val_loss:.4f} -> saved {best_ckpt_path}")

    history_path = os.path.join(args.checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    logger.info("=== Training run end ===")


if __name__ == "__main__":
    main()
