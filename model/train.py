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

Usage (Kaggle, full run — for reference, run there not here). Note that the
Kaggle inputs live under the read-only /kaggle/input tree, so the default
relative paths do not exist there and MUST be overridden:
    python model/train.py --subset-mode full --epochs 15 --batch-size 64 \
        --lr 1e-4 --num-workers 4 \
        --metadata /kaggle/input/<metadata-dataset>/metadata.csv \
        --image-dir /kaggle/input/<images-dataset>/images \
        --checkpoint-dir /kaggle/working/checkpoints \
        --log-path /kaggle/working/logs/train.log

--target log trains the head on log(price) instead of dollars (see
model/targets.py). The loss is then in log space and not comparable across
targets, so every epoch also logs MAE in dollars (`mae=$...`), which is.

Resuming after a Kaggle session timeout (attach the previous run's output as
a dataset, then point --resume-from at its checkpoints). Note that --epochs
is the FINAL epoch number, not a count of extra epochs:
    python model/train.py --subset-mode full --epochs 15 ... \
        --resume-from /kaggle/input/<previous-run>/checkpoints/epoch_7.pt \
        --checkpoint-dir /kaggle/working/checkpoints
"""
import argparse
import json
import logging
import os
import re
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import PriceImageDataset
from model.model import build_model, build_loss_fn, EXPECTED_INPUT_SIZE, BACKBONE_NAME
from model.targets import TARGETS, checkpoint_target, to_price, to_target


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
    # dirname("train.log") is "" — makedirs("") raises FileNotFoundError, so
    # only create a directory when the path actually names one.
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
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
              max_batches=None, log_every=50, target="price"):
    """One pass over `loader`. Returns (avg_loss, mae): the loss is in whatever
    space the model is trained in (dollars or log-dollars), the MAE is always
    in dollars, so runs with different targets can be compared on it."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_abs_err = 0.0
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
            loss = loss_fn(outputs, to_target(prices, target))

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_abs_err += (to_price(outputs.detach(), target) - prices).abs().sum().item()
            n_samples += images.size(0)
            n_batches += 1

            # Log every Nth batch rather than every batch: a full epoch is
            # ~800 batches at batch_size=64, and 15 epochs of per-batch lines
            # makes the Kaggle notebook output unusably large.
            if log_every > 0 and batch_idx % log_every == 0:
                logger.info(
                    f"epoch={epoch} split={split_name} batch={batch_idx} "
                    f"batch_loss={loss.item():.4f}"
                )

    elapsed = time.time() - start
    avg_loss = total_loss / max(n_samples, 1)
    mae = total_abs_err / max(n_samples, 1)
    logger.info(
        f"epoch={epoch} split={split_name} DONE avg_loss={avg_loss:.4f} mae=${mae:.3f} "
        f"batches={n_batches} samples={n_samples} elapsed_sec={elapsed:.1f}"
    )
    return avg_loss, mae


def history_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "history.json")


def save_history(checkpoint_dir: str, history) -> None:
    path = history_path(checkpoint_dir)
    with open(path + ".tmp", "w") as f:
        json.dump(history, f, indent=2)
    os.replace(path + ".tmp", path)


def save_checkpoint(checkpoint, path: str) -> None:
    # Write then rename, so a session killed mid-save can't leave a half-written
    # best.pt behind and wipe out the best model.
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_history(checkpoint_dir: str):
    path = history_path(checkpoint_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def resolve_resume_path(resume_from: str, checkpoint_dir: str) -> str:
    """Resolve --resume-from to a concrete checkpoint file.

    'latest' picks the highest-numbered epoch_N.pt in checkpoint_dir - the
    common case on Kaggle, where you re-attach the previous run's output and
    just want to carry on from wherever it died.
    """
    if resume_from != "latest":
        if not os.path.exists(resume_from):
            raise SystemExit(f"ERROR: --resume-from checkpoint not found at '{resume_from}'.")
        return resume_from

    names = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
    candidates = []
    for name in names:
        match = re.fullmatch(r"epoch_(\d+)\.pt", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))
    if not candidates:
        raise SystemExit(
            f"ERROR: --resume-from latest found no epoch_N.pt files in "
            f"'{checkpoint_dir}'. Point --checkpoint-dir at the directory holding the "
            f"previous run's checkpoints, or pass an explicit path."
        )
    return max(candidates)[1]


def main():
    parser = argparse.ArgumentParser(description="Train the price-regression model.")
    parser.add_argument("--subset-mode", choices=["dev", "full"], required=True,
                         help="Selects the default metadata/image paths. Override either "
                              "with --metadata / --image-dir (required on Kaggle, where "
                              "inputs live under the read-only /kaggle/input tree).")
    parser.add_argument("--metadata", default=None,
                         help="Override the metadata CSV path for the chosen subset mode.")
    parser.add_argument("--image-dir", default=None,
                         help="Override the resized-image directory for the chosen subset mode.")
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
    parser.add_argument("--resume-from", default=None,
                         help="Resume from a checkpoint written by a previous run. Either a "
                              "path to a .pt file, or the literal 'latest' to pick the "
                              "highest-numbered epoch_N.pt in --checkpoint-dir. Training "
                              "continues at the epoch after the one stored in the checkpoint, "
                              "and --epochs is still the FINAL epoch number, not a count of "
                              "additional epochs.")
    parser.add_argument("--log-every", type=int, default=50,
                         help="Log a batch_loss line every N batches (0 disables per-batch "
                              "logging). Epoch summaries are always logged.")
    parser.add_argument("--target", choices=TARGETS, default="price",
                         help="What the head is trained to predict: 'price' (dollars, run1) or "
                              "'log' (log of the price). See model/targets.py.")
    parser.add_argument("--select-by", choices=["val_loss", "val_mae"], default="val_loss",
                         help="Which val number decides best.pt. val_loss is in the target's "
                              "space (log-dollars with --target log); val_mae is always "
                              "dollars, so use it when comparing runs with different targets.")
    parser.add_argument("--max-hours", type=float, default=None,
                         help="Wall-clock budget for this run. Before each epoch, stop cleanly "
                              "if another epoch (at the average pace so far) would go over it. "
                              "Set it below the Kaggle session limit so the run finishes and "
                              "saves its output instead of being killed.")
    args = parser.parse_args()
    run_start = time.time()

    paths = dict(SUBSET_PATHS[args.subset_mode])
    if args.metadata:
        paths["metadata"] = args.metadata
    if args.image_dir:
        paths["image_dir"] = args.image_dir

    for label, path in (("metadata CSV", paths["metadata"]), ("image dir", paths["image_dir"])):
        if not os.path.exists(path):
            raise SystemExit(
                f"ERROR: {label} not found at '{path}'.\n"
                f"With --subset-mode {args.subset_mode} the default is "
                f"'{SUBSET_PATHS[args.subset_mode]['metadata' if label.startswith('metadata') else 'image_dir']}'. "
                f"On Kaggle, pass --metadata and --image-dir pointing into /kaggle/input/..."
            )

    logger = setup_logging(args.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Training run start ===")
    logger.info(f"subset_mode={args.subset_mode} device={device} "
                f"cuda_available={torch.cuda.is_available()}")
    logger.info(f"backbone={BACKBONE_NAME} input_size={EXPECTED_INPUT_SIZE} "
                f"target={args.target} epochs={args.epochs} batch_size={args.batch_size} "
                f"lr={args.lr} max_batches={args.max_batches}")
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
    if args.target == "log":
        for ds in (train_ds, val_ds):
            if (ds.df["price"] <= 0).any():
                raise SystemExit("ERROR: --target log needs every price > 0, found a price <= 0.")

    # pin_memory only helps (and only makes sense) when copying to a GPU.
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True,
                               pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(pretrained=args.pretrained).to(device)
    loss_fn = build_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []
    best_score = float("inf")  # lowest args.select_by seen so far
    start_epoch = 1

    if args.resume_from:
        ckpt_path = resolve_resume_path(args.resume_from, args.checkpoint_dir)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # The loss curve so far lives in the checkpoint (newer runs) or in
        # history.json next to it (checkpoints written before that was added).
        history = ckpt.get("history") or load_history(args.checkpoint_dir)
        history = [h for h in history if h["epoch"] < start_epoch]
        # Entries written before val_mae existed are skipped when selecting by it.
        best_score = min((h[args.select_by] for h in history if args.select_by in h),
                         default=float("inf"))

        # A different target isn't a hyperparameter tweak: the head's outputs
        # mean something else (dollars vs log-dollars), so refuse outright.
        if checkpoint_target(ckpt) != args.target:
            raise SystemExit(
                f"ERROR: checkpoint was trained with --target {checkpoint_target(ckpt)}, "
                f"but this run uses --target {args.target}. Resume with the same target."
            )

        prev_args = ckpt.get("args", {})
        for key in ("batch_size", "lr", "pretrained", "select_by"):
            if key in prev_args and prev_args[key] != getattr(args, key):
                logger.info(
                    f"WARNING: resuming with {key}={getattr(args, key)} but the checkpoint "
                    f"was written with {key}={prev_args[key]} - this run is not a "
                    f"continuation of the same hyperparameters."
                )
        logger.info(
            f"Resumed from {ckpt_path}: completed_epoch={ckpt['epoch']} "
            f"val_loss={ckpt.get('val_loss')} -> starting at epoch {start_epoch}, "
            f"best {args.select_by} so far={best_score:.4f}"
        )
        if start_epoch > args.epochs:
            raise SystemExit(
                f"ERROR: checkpoint is already at epoch {ckpt['epoch']} but --epochs is "
                f"{args.epochs}. --epochs is the final epoch number, so raise it above "
                f"{ckpt['epoch']} to train further."
            )

    epoch_times = []
    for epoch in range(start_epoch, args.epochs + 1):
        if args.max_hours is not None and epoch_times:
            elapsed = time.time() - run_start
            avg_epoch = sum(epoch_times) / len(epoch_times)
            if elapsed + avg_epoch > args.max_hours * 3600:
                logger.info(
                    f"STOPPING EARLY before epoch {epoch}: {elapsed / 3600:.2f} h used, "
                    f"next epoch needs ~{avg_epoch / 60:.0f} min, budget is "
                    f"{args.max_hours} h. Continue later with --resume-from."
                )
                break

        epoch_start = time.time()
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                           logger, epoch, "train", max_batches=args.max_batches,
                                           log_every=args.log_every, target=args.target)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn, None, device,
                                       logger, epoch, "val", max_batches=args.max_batches,
                                       log_every=args.log_every, target=args.target)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_mae": train_mae, "val_mae": val_mae})

        # Save a checkpoint every epoch (so a Kaggle run can resume after an
        # interruption), plus best.pt for the best epoch by --select-by.
        epoch_ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "args": vars(args),
            # Carried so --resume-from restores the full loss curve and the
            # best-so-far watermark without depending on history.json.
            "history": history,
        }
        save_checkpoint(checkpoint, epoch_ckpt_path)
        logger.info(f"Saved checkpoint: {epoch_ckpt_path}")

        # Rewrite history.json every epoch, not just at the end: a Kaggle
        # session that times out mid-run would otherwise leave no loss curve.
        save_history(args.checkpoint_dir, history)

        score = history[-1][args.select_by]
        if score < best_score:
            best_score = score
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
            save_checkpoint(checkpoint, best_ckpt_path)
            logger.info(f"New best {args.select_by}={score:.4f} -> saved {best_ckpt_path}")

        epoch_times.append(time.time() - epoch_start)
        logger.info(f"epoch={epoch} took {epoch_times[-1] / 60:.1f} min, "
                    f"run total {(time.time() - run_start) / 3600:.2f} h")

    scored = [h for h in history if args.select_by in h]
    if scored:
        best = min(scored, key=lambda h: h[args.select_by])
        best_mae = f" val_mae=${best['val_mae']:.3f}" if "val_mae" in best else ""
        logger.info(f"best_epoch={best['epoch']} (by {args.select_by}) "
                    f"best_val_loss={best['val_loss']:.4f}{best_mae} "
                    f"last_epoch={history[-1]['epoch']}")
    logger.info(f"Saved training history to {history_path(args.checkpoint_dir)}")
    logger.info("=== Training run end ===")


if __name__ == "__main__":
    main()
