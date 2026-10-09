"""
Download the model the API serves from the GitHub release, and check it.

    python backend/download_model.py
    python backend/download_model.py --from-file path/to/price_model.onnx

Saves price_model.onnx and price_model.json into backend/models/ (ignored by
git). The sha256 below is the one recorded on Kaggle when the model was
exported and verified, so a damaged or different file is rejected rather
than served. Running it again does nothing if the files are already correct.

--from-file installs a copy you already have (its price_model.json must sit
next to it), after the same check. Useful when GitHub's download server is
unreachable: some networks reset connections to release-assets.githubusercontent.com.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RELEASE = "v0.1-model"
BASE_URL = f"https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor/releases/download/{RELEASE}/"
EXPECTED_SHA256 = "b7055048f0f059789f740e121c7b739c67b2269845c2e90814273f6cccebf984"
MODELS = Path(__file__).resolve().parent / "models"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, attempts=3):
    target = MODELS / name
    partial = target.with_name(name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response, \
                    open(partial, "wb") as out:
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total > 10 * 1024 * 1024:
                        print(f"\r  {name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            if total > 10 * 1024 * 1024:
                print()
            os.replace(partial, target)  # only a complete download gets the real name
            return target
        except (urllib.error.URLError, OSError) as e:
            print(f"  {name}: attempt {attempt} of {attempts} failed ({e})")
            if attempt < attempts:
                time.sleep(3 * attempt)
    partial.unlink(missing_ok=True)
    sys.exit(
        f"ERROR: couldn't download {name} from GitHub. Check your connection and try again.\n"
        f"If it keeps failing, your network may be blocking GitHub's download server. Download "
        f"both files from https://github.com/Akash7762/Amazon-Vision-Based-Price-Predictor/releases/tag/{RELEASE}\n"
        f"in a browser (or on another network), then run:\n"
        f"  python backend/download_model.py --from-file path/to/price_model.onnx")


def check(model, card):
    actual = sha256(model)
    recorded = json.loads(Path(card).read_text())["sha256"]
    if actual != EXPECTED_SHA256 or recorded != EXPECTED_SHA256:
        sys.exit(f"ERROR: {model} has sha256 {actual}, expected {EXPECTED_SHA256} ({RELEASE}). "
                 "It's damaged or a different model.")


def main():
    ap = argparse.ArgumentParser(description=f"Get the {RELEASE} model for the API.")
    ap.add_argument("--from-file", metavar="ONNX",
                    help="install a local price_model.onnx (with price_model.json next to it)")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    model = MODELS / "price_model.onnx"
    card = MODELS / "price_model.json"

    if args.from_file:
        source = Path(args.from_file)
        source_card = source.with_suffix(".json")
        if not source.exists() or not source_card.exists():
            sys.exit(f"ERROR: need both {source} and {source_card}.")
        check(source, source_card)
        shutil.copyfile(source_card, card)
        shutil.copyfile(source, model)
        print(f"OK: installed {model} from {source}; sha256 matches {RELEASE}.")
        return

    if model.exists() and card.exists() and sha256(model) == EXPECTED_SHA256:
        print(f"{model} is already the {RELEASE} model. Nothing to do.")
        return

    print(f"Downloading {RELEASE} from GitHub...")
    download("price_model.json")
    download("price_model.onnx")
    try:
        check(model, card)
    except SystemExit:
        model.unlink()
        raise
    print(f"OK: {model} ({model.stat().st_size / 1e6:.1f} MB), sha256 matches {RELEASE}.")


if __name__ == "__main__":
    main()
