"""
What the regression head is trained to predict.

    "price"  the head outputs dollars directly (run1).
    "log"    the head outputs log(price), and dollars are exp(output).

Why log: prices here run from $1.32 to $145 with a long right tail. On the
raw scale the loss is dominated by expensive items, since being $20 off a
$100 item costs as much as being $20 off a $5 item. On the log scale an error
is a ratio: predicting $4 for a $2 item and $100 for a $50 item are both
"2x too high" and cost the same. exp() also can't produce a negative price.

Plain log rather than log1p is safe because the cleaned data has no price
below $1.32 (Phase 1 clips to the 1st-99th percentile), and it keeps that
"error = ratio" reading exact.

Anything that trains with a transform has to undo it before reporting
dollars, so training, evaluation and the Phase 4 backend all go through
these two functions.
"""
import torch

TARGETS = ("price", "log")


def to_target(price: torch.Tensor, kind: str) -> torch.Tensor:
    """Dollar prices -> what the model is trained on."""
    if kind == "price":
        return price
    if kind == "log":
        return torch.log(price)
    raise ValueError(f"unknown target {kind!r}, expected one of {TARGETS}")


def to_price(output: torch.Tensor, kind: str) -> torch.Tensor:
    """Model output -> dollars."""
    if kind == "price":
        return output
    if kind == "log":
        return torch.exp(output)
    raise ValueError(f"unknown target {kind!r}, expected one of {TARGETS}")


def checkpoint_target(checkpoint: dict) -> str:
    """The target a checkpoint was trained with. run1 predates the option, so
    a checkpoint without it was trained on raw price."""
    return checkpoint.get("args", {}).get("target", "price")
