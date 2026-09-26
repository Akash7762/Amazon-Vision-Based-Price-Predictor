"""
Price-regression model definition (Phase 2).

Backbone: convnext_tiny.fb_in22k_ft_in1k (via timm)
----------------------------------------------------
Chosen over EfficientNetV2-S variants for one concrete, verified reason:
its native pretrained input resolution is 224x224, which matches the
resize-at-download decision already locked in for this project (images are
downloaded and cached at 224x224, not at original resolution). This was
confirmed by inspecting the installed timm 1.0.29's actual pretrained
config rather than assumed:

    >>> import timm
    >>> timm.get_pretrained_cfg('convnext_tiny.fb_in22k_ft_in1k').input_size
    (3, 224, 224)

By contrast, the EfficientNetV2 variants available in this timm version
(`tf_efficientnetv2_s`: 300x300, `efficientnetv2_rw_s`: 288x288) do NOT
natively expect 224x224 input — using them would mean either upscaling our
already-downsized 224x224 cached images (losing the point of matching
native resolution) or re-deciding the cache resolution, which is out of
scope for this phase. convnext_tiny.fb_in22k_ft_in1k avoids that conflict
entirely while still being a strong, modern, ImageNet-22k-then-1k
pretrained backbone (~28.6M params).

Regression head
---------------
timm's `create_model(..., num_classes=1)` already replaces the backbone's
classifier with a single `nn.Linear(in_features, 1)` layer when
num_classes=1 — this *is* the regression head (single scalar output, no
activation, i.e. it predicts raw price directly rather than a class
distribution). No extra head code is needed on top of what timm provides;
this was verified by an actual forward pass (see docs/kaggle_training_guide.md
/ the Phase 2 run log) producing an output tensor of shape (batch_size, 1).
"""
import timm
import torch.nn as nn

BACKBONE_NAME = "convnext_tiny.fb_in22k_ft_in1k"
EXPECTED_INPUT_SIZE = 224  # verified via timm.get_pretrained_cfg(BACKBONE_NAME).input_size


def build_model(pretrained: bool = True) -> nn.Module:
    """Build the price-regression model: convnext_tiny backbone + single-output
    linear regression head (via timm's num_classes=1 mechanism)."""
    model = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=1)
    return model


def build_loss_fn() -> nn.Module:
    """Loss function for price regression: SmoothL1Loss (Huber loss).

    Justification: raw MSE penalizes large errors quadratically, which makes
    training oversensitive to whatever price outliers remain even after the
    Phase 1 1st/99th-percentile clipping (valid range is still wide: $1.32
    to $145.25, and mid/high-price items will naturally produce larger
    absolute errors early in training). SmoothL1 (Huber) behaves like MSE
    for small errors (smooth gradient near zero) but like MAE (linear, less
    dominated by outliers) for large errors, which is the standard choice
    for regression targets with a long-tailed distribution like this one.
    """
    return nn.SmoothL1Loss()
