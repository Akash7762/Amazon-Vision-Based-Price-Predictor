"""
Export a trained checkpoint to ONNX for the backend (Phase 3), and prove the
exported file gives the same prices as PyTorch.

    python model/export.py --checkpoint .../checkpoints/best.pt --out-dir export \
        --metadata .../metadata.csv --image-dir .../images --verify-split test

Writes to --out-dir:
    price_model.onnx   the model, a single file (~110 MB)
    price_model.json   model card: input/output format, source checkpoint,
                       sha256, and the verification result

Why ONNX: onnxruntime runs it without PyTorch, so the Phase 4 backend stays
small (TorchScript would tie it to a full torch install).

What the exported graph does, so the backend can't get it subtly wrong:

    input  "image"  uint8 (1, 224, 224, 3), RGB, the photo already letterboxed
                    to 224x224 with model.preprocessing.letterbox_resize
      -> scale to [0, 1], ImageNet normalisation, channels first
      -> ConvNeXt-Tiny + regression head
      -> back to dollars (exp() for a log-price model)
    output "price"  float32 (1,), US dollars

Normalisation is inside the graph on purpose: done wrong in the backend it
fails silently, with no error, just worse prices. The backend only letterboxes
the photo and passes the raw pixels.

Verification feeds the same test images through two independent routes:
PyTorch with the exact evaluation transform from model/dataset.py, and
onnxruntime with raw letterboxed pixels as the backend will send them. The
export only counts as done if every price agrees within --tolerance dollars.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import IMAGENET_MEAN, IMAGENET_STD, PriceImageDataset
from model.model import BACKBONE_NAME, EXPECTED_INPUT_SIZE, build_model
from model.preprocessing import letterbox_resize
from model.targets import checkpoint_target, to_price


class PriceModel(nn.Module):
    """uint8 RGB pixels (N, H, W, 3) in, dollars (N,) out."""

    def __init__(self, backbone, target):
        super().__init__()
        self.backbone = backbone
        self.target = target
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        # Same operations, in the same order, as ToTensor() + Normalize().
        x = image.permute(0, 3, 1, 2).float().div(255.0)
        x = (x - self.mean) / self.std
        return to_price(self.backbone(x), self.target).squeeze(1)


def export_onnx(model, path):
    """Try the current exporter, then the older TorchScript-based one."""
    example = (torch.zeros(1, EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE, 3, dtype=torch.uint8),)
    params = inspect.signature(torch.onnx.export).parameters
    attempts = []
    if "dynamo" in params:
        attempts.append(("dynamo", {"dynamo": True}))
        attempts.append(("torchscript", {"dynamo": False, "opset_version": 17}))
    else:
        attempts.append(("torchscript", {"opset_version": 17}))

    errors = []
    for label, extra in attempts:
        kwargs = dict(input_names=["image"], output_names=["price"], **extra)
        if "external_data" in params:
            kwargs["external_data"] = False  # one file, not .onnx + .onnx.data
        try:
            torch.onnx.export(model, example, path, **kwargs)
            return label
        except Exception as e:  # noqa: BLE001 - report every attempt, then give up
            errors.append(f"{label}: {type(e).__name__}: {e}")
    raise SystemExit("ERROR: ONNX export failed.\n" + "\n".join(errors))


def single_file(path):
    """Fold any external-data sidecar back into the .onnx (older exporters)."""
    import onnx
    folder, base = os.path.split(os.path.abspath(path))
    sidecars = [f for f in os.listdir(folder) if f.startswith(base) and f != base]
    if sidecars:
        onnx.save_model(onnx.load(path), path, save_as_external_data=False)
        for f in sidecars:
            os.remove(os.path.join(folder, f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def verify(backbone, target, onnx_path, metadata, image_dir, split, n, tolerance, seed=0):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ds = PriceImageDataset(metadata, image_dir, split=split,
                           image_size=EXPECTED_INPUT_SIZE, train=False)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)

    diffs, refs = [], []
    for i in idx:
        x, _ = ds[int(i)]  # PyTorch route: the evaluation transform
        ref = float(to_price(backbone(x.unsqueeze(0)), target).item())
        sid = ds.df.iloc[int(i)]["sample_id"]  # ONNX route: raw pixels, as the backend sends them
        img = letterbox_resize(Image.open(os.path.join(image_dir, f"{sid}.jpg")),
                               EXPECTED_INPUT_SIZE)
        out = float(sess.run(["price"], {"image": np.asarray(img, dtype=np.uint8)[None]})[0][0])
        diffs.append(abs(out - ref))
        refs.append(abs(ref))
    diffs = np.array(diffs)
    return {
        "split": split,
        "n_images": int(len(idx)),
        "max_abs_diff_usd": float(diffs.max()),
        "mean_abs_diff_usd": float(diffs.mean()),
        "max_rel_diff": float((diffs / np.maximum(refs, 1e-6)).max()),
        "tolerance_usd": tolerance,
        "passed": bool(diffs.max() <= tolerance),
    }


def main():
    ap = argparse.ArgumentParser(description="Export a checkpoint to ONNX and verify it.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="export")
    ap.add_argument("--name", default="price_model")
    ap.add_argument("--metadata", required=True, help="for verification images")
    ap.add_argument("--image-dir", required=True, help="for verification images")
    ap.add_argument("--verify-split", default="test")
    ap.add_argument("--verify-n", type=int, default=256)
    ap.add_argument("--tolerance", type=float, default=0.01, help="max allowed difference, $")
    args = ap.parse_args()

    import onnx
    import onnxruntime

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target = checkpoint_target(ckpt)
    backbone = build_model(pretrained=False)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.eval()
    model = PriceModel(backbone, target).eval()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.name}.onnx")
    exporter = export_onnx(model, path)
    single_file(path)
    onnx.checker.check_model(path)
    print(f"Exported with the {exporter} exporter -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")

    result = verify(backbone, target, path, args.metadata, args.image_dir,
                    args.verify_split, args.verify_n, args.tolerance)
    print(f"Verification on {result['n_images']} {result['split']} images: "
          f"max difference ${result['max_abs_diff_usd']:.6f}, "
          f"mean ${result['mean_abs_diff_usd']:.6f} "
          f"-> {'PASSED' if result['passed'] else 'FAILED'} (tolerance ${args.tolerance})")

    card = {
        "file": os.path.basename(path),
        "sha256": sha256(path),
        "size_mb": round(os.path.getsize(path) / 1e6, 1),
        "exporter": exporter,
        "opset": max(o.version for o in onnx.load(path, load_external_data=False).opset_import
                     if o.domain in ("", "ai.onnx")),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": onnxruntime.__version__},
        "source_checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "target": target,
        "backbone": BACKBONE_NAME,
        "input": {"name": "image", "dtype": "uint8",
                  "shape": [1, EXPECTED_INPUT_SIZE, EXPECTED_INPUT_SIZE, 3],
                  "layout": "NHWC, RGB",
                  "preprocessing": "model.preprocessing.letterbox_resize(img, 224): keep the "
                                   "aspect ratio, pad to a square with white"},
        "output": {"name": "price", "dtype": "float32", "shape": [1], "units": "USD"},
        "verification": result,
    }
    card_path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"Model card -> {card_path}")
    if not result["passed"]:
        raise SystemExit("ERROR: exported model disagrees with PyTorch beyond the tolerance.")


if __name__ == "__main__":
    main()
