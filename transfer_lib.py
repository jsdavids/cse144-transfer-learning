"""Shared transfer-learning library for the CSE 144 final project.

This module factors out everything the per-model notebooks/scripts used to
duplicate: numeric-labelled dataset construction, the train/val split,
augmentation (RandomRotation + MixUp/CutMix with soft labels), the fine-tuning
loop with AMP, training-curve plotting, and test-set inference.

A model is described by a small entry in ``MODEL_REGISTRY`` (which torchvision
backbone, which weights, where the classifier head lives, the backbone LR, and
the native input size). Adding a new architecture is a few lines there — no new
copy of the training code.

Design notes / correctness invariants (carried over verbatim from the original
resnet.ipynb / vit.ipynb so behaviour does not drift):

* **Numeric label ordering.** Class folders are named "0".."99". ImageFolder
  sorts them lexicographically ("0","1","10",...), which scrambles labels. We
  force folder "k" -> label k.
* **Submission length.** test/ has 1036 images but Kaggle scores only IDs
  0..999. ``run_inference(..., predict_all=True)`` writes all 1036 and verifies
  the 1000 scored IDs are covered.
* **Soft-label loss.** MixUp/CutMix produce two targets + a mixing weight; the
  training loss is ``lam*CE(y_a) + (1-lam)*CE(y_b)`` using the label-smoothed
  criterion. Validation uses plain CE on the real labels.
* **Best-checkpoint-on-val-accuracy.** The saved .pth is always the best epoch
  and stores ``class_names`` + ``val_acc`` so inference is self-describing.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import models as tv_models
from torchvision import transforms
from torchvision.datasets import ImageFolder

# --------------------------------------------------------------------------- #
# Constants shared across all models
# --------------------------------------------------------------------------- #
SEED = 42
NUM_CLASSES = 100
VAL_FRAC = 0.2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BASELINE_ACC = 0.60  # Kaggle baseline to clear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
USE_AMP = USE_CUDA
if USE_CUDA:
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# Each entry says how to build a backbone and where its classifier head lives.
#   builder      : the torchvision constructor (called with weights=...)
#   weights      : the pretrained weights enum (its .DEFAULT is used)
#   head_path    : dotted path to the final Linear; integer segments index into
#                  nn.Sequential (e.g. "classifier.-1" -> model.classifier[-1])
#   lr_backbone  : per-arch backbone LR (transformers like a smaller value)
#   img_size     : native input resolution (efficientnet_b3 wants 300, not 224)
@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., nn.Module]
    weights: object
    head_path: str
    lr_backbone: float
    img_size: int = 224


def _w(name: str):
    """Look up a torchvision weights enum by attribute name, return its DEFAULT."""
    return getattr(tv_models, name).DEFAULT


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "resnet152": ModelSpec(
        tv_models.resnet152, _w("ResNet152_Weights"), "fc", lr_backbone=1e-4),
    "resnet50": ModelSpec(
        tv_models.resnet50, _w("ResNet50_Weights"), "fc", lr_backbone=1e-4),
    "vit_b16": ModelSpec(
        tv_models.vit_b_16, _w("ViT_B_16_Weights"), "heads.head", lr_backbone=1e-5),
    "convnext_small": ModelSpec(
        tv_models.convnext_small, _w("ConvNeXt_Small_Weights"),
        "classifier.-1", lr_backbone=1e-4),
    "swin_s": ModelSpec(
        tv_models.swin_s, _w("Swin_S_Weights"), "head", lr_backbone=1e-5),
    "efficientnet_b3": ModelSpec(
        tv_models.efficientnet_b3, _w("EfficientNet_B3_Weights"),
        "classifier.-1", lr_backbone=1e-4, img_size=300),
}


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    model_name: str
    epochs: int = 30
    fine_tune_mode: str = "full"           # "full" | "layer4"/"last_block" | "head"
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    dropout: float = 0.0                    # >0 inserts Dropout(p) before the head
    # Gradual unfreezing: start with the backbone frozen (head only), then thaw
    # one block (head-end first) every `unfreeze_every` epochs. 0 disables it
    # (the backbone follows fine_tune_mode from epoch 1, the original behaviour).
    gradual_unfreeze: bool = False
    unfreeze_every: int = 2                 # epochs between thawing successive blocks
    # Augmentation
    use_rotation: bool = True
    rotation_deg: int = 20
    use_mixup_cutmix: bool = True
    mix_prob: float = 0.5
    mix_alpha: float = 0.2
    # RandAugment: when on, it REPLACES the per-image rotation + ColorJitter
    # (it already samples rotate/shear/color/contrast/etc.), so we don't stack
    # and over-regularize. MixUp/CutMix (batch-level) are orthogonal and stay.
    use_randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9          # torchvision default; lower for less aggression
    # Test-time augmentation (inference only)
    use_tta: bool = False
    tta_hflip: bool = True                  # add the horizontal-flip view
    tta_five_crop: bool = False             # add 4 corner + center crops (5 views)
    # Data / runtime
    seed: int = SEED                        # controls val split + weight init; vary to gauge noise
    data_root: Path = field(default_factory=lambda: Path("ucsc-cse-144-spring-2026-final-project"))
    num_workers: int = 0                    # 0 is required on Windows (see README)
    batch_size: int | None = None           # None -> 16 on CUDA, 8 on CPU

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return 16 if USE_CUDA else 8

    @property
    def _run_suffix(self) -> str:
        """Tags output files so runs that differ in a gap-affecting way don't
        overwrite each other: '_seed<N>' for non-default seeds, '_gu' when
        gradual unfreezing is on, '_ra' when RandAugment is on. The default
        seed-42 full-fine-tune run keeps the canonical bare names (and the
        existing checkpoints + ensemble)."""
        suffix = "" if self.seed == SEED else f"_seed{self.seed}"
        if self.gradual_unfreeze:
            suffix += "_gu"
        if self.use_randaugment:
            suffix += "_ra"
        return suffix

    @property
    def spec(self) -> ModelSpec:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {self.model_name!r}. "
                f"Choices: {sorted(MODEL_REGISTRY)}")
        return MODEL_REGISTRY[self.model_name]

    @property
    def ckpt_path(self) -> str:
        return f"{self.model_name}_finetuned{self._run_suffix}.pth"

    @property
    def submission_path(self) -> str:
        return f"submission_{self.model_name}{self._run_suffix}.csv"

    @property
    def curves_path(self) -> str:
        return f"{self.model_name}_training_curves{self._run_suffix}.png"


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(cfg: Config):
    """Return (train_tfms, eval_tfms) for this model's input size."""
    img_size = cfg.spec.img_size
    resize = int(round(img_size * 256 / 224))  # keep the 256/224 ratio at any res

    train_ops = []
    if cfg.use_randaugment:
        # RandAugment subsumes rotation + ColorJitter (it samples rotate/shear/
        # color/contrast/brightness/posterize/...), so it REPLACES them rather
        # than stacking. It runs first, before the crop hides any black corners
        # its geometric ops introduce.
        train_ops.append(transforms.RandAugment(
            num_ops=cfg.randaugment_num_ops, magnitude=cfg.randaugment_magnitude))
        train_ops += [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        if cfg.use_rotation:
            # Rotation goes BEFORE the random crop so the crop hides the rotated
            # black corners.
            train_ops.append(transforms.RandomRotation(cfg.rotation_deg))
        train_ops += [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_tfms = transforms.Compose(train_ops)

    eval_tfms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms


# --------------------------------------------------------------------------- #
# Dataset (numeric label ordering)
# --------------------------------------------------------------------------- #
def _numeric_class_mapping(train_dir: Path):
    class_names = sorted(os.listdir(train_dir), key=int)      # ["0",...,"99"]
    class_to_idx = {name: int(name) for name in class_names}  # folder "k" -> k
    return class_names, class_to_idx


def make_dataset(train_dir: Path, tfms, class_to_idx):
    """ImageFolder with the lexicographic mapping overridden to be numeric."""
    ds = ImageFolder(str(train_dir), transform=tfms)
    old_idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    ds.samples = [(p, class_to_idx[old_idx_to_class[t]]) for p, t in ds.samples]
    ds.targets = [t for _, t in ds.samples]
    ds.imgs = ds.samples
    ds.class_to_idx = class_to_idx
    ds.classes = sorted(class_to_idx, key=int)
    return ds


def build_loaders(cfg: Config):
    """Build train/val DataLoaders + return (class_names, train_loader, val_loader)."""
    set_seed(cfg.seed)
    train_dir = cfg.data_root / "train"
    class_names, class_to_idx = _numeric_class_mapping(train_dir)
    train_tfms, eval_tfms = build_transforms(cfg)

    # Sanity check: a file in folder "10" must carry label 10.
    check = make_dataset(train_dir, eval_tfms, class_to_idx)
    ex = next(p for p, _ in check.samples
              if os.path.sep + "10" + os.path.sep in p)
    assert dict(check.samples)[ex] == 10, "Label ordering is wrong!"

    n = len(check)
    indices = list(range(n))
    random.Random(cfg.seed).shuffle(indices)
    val_size = int(round(VAL_FRAC * n))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = Subset(make_dataset(train_dir, train_tfms, class_to_idx), train_idx)
    val_ds = Subset(make_dataset(train_dir, eval_tfms, class_to_idx), val_idx)

    bs = cfg.resolved_batch_size()
    pin = USE_CUDA
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    return class_names, train_loader, val_loader, eval_tfms


# --------------------------------------------------------------------------- #
# Model factory (generic head swap via dotted path)
# --------------------------------------------------------------------------- #
def _resolve_parent_and_key(model: nn.Module, head_path: str):
    """Walk a dotted path, returning (parent_module, final_key) so the final
    segment can be replaced. Integer segments index into a Sequential."""
    parts = head_path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.lstrip("-").isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _get_head(model: nn.Module, head_path: str) -> nn.Linear:
    parent, key = _resolve_parent_and_key(model, head_path)
    return parent[int(key)] if key.lstrip("-").isdigit() else getattr(parent, key)


def _set_head(model: nn.Module, head_path: str, new_head: nn.Module) -> None:
    parent, key = _resolve_parent_and_key(model, head_path)
    if key.lstrip("-").isdigit():
        parent[int(key)] = new_head
    else:
        setattr(parent, key, new_head)


def _backbone_block_for_mode(model: nn.Module, model_name: str):
    """Return the 'last block' submodule to unfreeze for the partial mode, or
    None if this arch doesn't define an obvious one."""
    if hasattr(model, "layer4"):                 # ResNet family
        return model.layer4
    if hasattr(model, "features"):               # ConvNeXt / EfficientNet
        return model.features[-1]
    if hasattr(model, "encoder"):                # ViT
        return model.encoder
    return None


def _backbone_blocks_top_down(model: nn.Module) -> list[nn.Module]:
    """Ordered list of backbone blocks from the head end down to the stem.

    Gradual unfreezing walks this list front-to-back, so index 0 (the block
    closest to the classifier) thaws first and the input stem thaws last. The
    classifier head itself is NOT included — it is always trainable.
    """
    if hasattr(model, "layer4"):                 # ResNet: stages + input stem
        blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
        stem = [m for m in (getattr(model, "conv1", None), getattr(model, "bn1", None))
                if m is not None]
        return blocks + ([nn.ModuleList(stem)] if stem else [])
    if hasattr(model, "features"):               # ConvNeXt / EfficientNet / Swin
        return [model.features[i] for i in range(len(model.features) - 1, -1, -1)]
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):  # ViT
        enc = model.encoder
        blocks = [enc.layers[i] for i in range(len(enc.layers) - 1, -1, -1)]
        # patch/conv-proj + class token + pos-embed live on the model itself;
        # thaw the remaining encoder norm + projection last as one group.
        tail = [m for m in (getattr(enc, "ln", None),
                            getattr(model, "conv_proj", None)) if m is not None]
        return blocks + ([nn.ModuleList(tail)] if tail else [])
    return []


def build_model(cfg: Config) -> nn.Module:
    spec = cfg.spec
    model = spec.builder(weights=spec.weights)

    mode = cfg.fine_tune_mode
    if cfg.gradual_unfreeze:
        # Start head-only; train() thaws backbone blocks on a schedule. We freeze
        # everything here and let the head swap below re-enable the classifier.
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode in ("layer4", "last_block"):
        for p in model.parameters():
            p.requires_grad = False
        block = _backbone_block_for_mode(model, cfg.model_name)
        if block is None:
            raise ValueError(
                f"Partial fine-tune not defined for {cfg.model_name!r}; "
                f"use 'full' or 'head'.")
        for p in block.parameters():
            p.requires_grad = True
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unknown fine_tune_mode: {mode!r}")

    # Swap the ImageNet head for a NUM_CLASSES-way Linear (always trainable).
    # With dropout>0, prepend a Dropout so the new head is regularized — this is
    # the main lever against overfitting on ~10 images/class. The wrapping
    # Sequential keeps the same parent attribute name, so build_optimizer's
    # head-prefix split still routes these params to the head LR group.
    in_features = _get_head(model, spec.head_path).in_features
    new_head = nn.Linear(in_features, NUM_CLASSES)
    if cfg.dropout > 0:
        new_head = nn.Sequential(nn.Dropout(cfg.dropout), new_head)
    _set_head(model, spec.head_path, new_head)

    # Ensure the ENTIRE head-prefix subtree is trainable, not just the swapped
    # Linear. Some classifiers are Sequentials with a pre-head norm/pool (e.g.
    # ConvNeXt's classifier[0] LayerNorm); under gradual unfreeze / head mode the
    # blanket freeze above would otherwise strand those, since they belong to the
    # head group (build_optimizer) but live in no unfreeze block.
    head_prefix = spec.head_path.split(".")[0]
    for name, p in model.named_parameters():
        if name.startswith(head_prefix):
            p.requires_grad = True

    return model.to(DEVICE)


def build_optimizer(cfg: Config, model: nn.Module):
    """Two LR groups: head at lr_head, backbone tensors at the arch's lr_backbone.

    Normally only currently-trainable tensors are added. With gradual unfreezing
    the backbone group must include *all* backbone tensors up front so AdamW has
    them registered (with their LR + optimizer state) for when train() thaws
    them later — a frozen tensor simply receives no gradient until then.
    """
    head_prefix = cfg.spec.head_path.split(".")[0]  # e.g. "fc", "classifier", "heads"
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        is_head = name.startswith(head_prefix)
        if is_head:
            if p.requires_grad:
                head_params.append(p)
        elif p.requires_grad or cfg.gradual_unfreeze:
            backbone_params.append(p)

    groups = [{"params": head_params, "lr": cfg.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.spec.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return optimizer, scheduler, len(head_params), len(backbone_params)


# --------------------------------------------------------------------------- #
# MixUp / CutMix + soft-label loss
# --------------------------------------------------------------------------- #
def _rand_bbox(h, w, lam):
    """Random box whose area is (1 - lam) of the image (for CutMix)."""
    cut_rat = np.sqrt(1.0 - lam)
    cw, ch = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def mix_batch(imgs, labels, cfg: Config):
    """Return (mixed_imgs, labels_a, labels_b, lam).

    With prob (1 - mix_prob) returns the batch unchanged (lam=1). Otherwise
    applies MixUp or CutMix (50/50) using a shuffled copy of the batch as the
    second example.
    """
    if (not cfg.use_mixup_cutmix) or (np.random.rand() > cfg.mix_prob):
        return imgs, labels, labels, 1.0

    lam = float(np.random.beta(cfg.mix_alpha, cfg.mix_alpha))
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    labels_b = labels[perm]

    if np.random.rand() < 0.5:                      # ---- MixUp ----
        imgs = lam * imgs + (1.0 - lam) * imgs[perm]
    else:                                           # ---- CutMix ----
        h, w = imgs.shape[2], imgs.shape[3]
        x1, y1, x2, y2 = _rand_bbox(h, w, lam)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    return imgs, labels, labels_b, lam


def make_soft_ce(criterion):
    def soft_ce(logits, labels_a, labels_b, lam):
        return (lam * criterion(logits, labels_a)
                + (1.0 - lam) * criterion(logits, labels_b))
    return soft_ce


# --------------------------------------------------------------------------- #
# Gradual unfreezing
# --------------------------------------------------------------------------- #
def _maybe_thaw(cfg: Config, model, blocks, n_thawed: int, epoch: int, *, verbose):
    """Thaw the next backbone block(s) whose scheduled epoch has arrived.

    Block i (head-end first) thaws at epoch 1 + i*unfreeze_every. Returns the
    updated count of thawed blocks. Params already in the optimizer's backbone
    group (see build_optimizer) start training the moment requires_grad flips on.
    """
    if not cfg.gradual_unfreeze:
        return n_thawed
    while n_thawed < len(blocks) and epoch >= 1 + n_thawed * cfg.unfreeze_every:
        for p in blocks[n_thawed].parameters():
            p.requires_grad = True
        n_thawed += 1
        if verbose:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [gradual] thawed block {n_thawed}/{len(blocks)} "
                  f"-> {n_train:,} trainable params")
    return n_thawed


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train(cfg: Config, model, train_loader, val_loader, class_names, *, verbose=True):
    """Fine-tune, saving the best-val checkpoint. Returns (history, best_val_acc).

    history rows are (train_loss, train_acc, val_loss, val_acc).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    soft_ce = make_soft_ce(criterion)
    optimizer, scheduler, n_head, n_back = build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    if verbose:
        print(f"Optimizing {n_head} head tensors + {n_back} backbone tensors")

    blocks = _backbone_blocks_top_down(model) if cfg.gradual_unfreeze else []
    n_thawed = 0
    if cfg.gradual_unfreeze and verbose:
        last = 1 + (len(blocks) - 1) * cfg.unfreeze_every
        print(f"Gradual unfreeze: {len(blocks)} blocks, one every "
              f"{cfg.unfreeze_every} epoch(s); fully unfrozen by epoch {last}")

    def run_epoch(loader, training: bool):
        model.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        torch.set_grad_enabled(training)
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
                mixed, lab_a, lab_b, lam = mix_batch(imgs, labels, cfg)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    out = model(mixed)
                    loss = soft_ce(out, lab_a, lab_b, lam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ref = lab_a if lam >= 0.5 else lab_b
                correct += (out.argmax(1) == ref).sum().item()
            else:
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    out = model(imgs)
                    loss = criterion(out, labels)
                correct += (out.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * imgs.size(0)
            total += imgs.size(0)
        torch.set_grad_enabled(True)
        return loss_sum / total, correct / total

    best_val_acc = 0.0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        n_thawed = _maybe_thaw(cfg, model, blocks, n_thawed, epoch, verbose=verbose)
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(val_loader, False)
        scheduler.step()
        history.append((tr_loss, tr_acc, va_loss, va_acc))
        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names,
                        "val_acc": va_acc,
                        "model_name": cfg.model_name,
                        "dropout": cfg.dropout,
                        "seed": cfg.seed,
                        "gradual_unfreeze": cfg.gradual_unfreeze,
                        "best_epoch": epoch}, cfg.ckpt_path)
            flag = "  <- saved best"
        if verbose:
            print(f"Epoch {epoch:2d}/{cfg.epochs} | "
                  f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | "
                  f"{time.time()-t0:5.1f}s{flag}")
    if verbose:
        print(f"\nBest val accuracy: {best_val_acc:.3f}")
    return history, best_val_acc


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def plot_curves(cfg: Config, history, *, show=True):
    import matplotlib.pyplot as plt

    if not history:
        print("No training history — run train() first.")
        return
    hist = np.array(history, dtype=float)
    epochs = np.arange(1, len(hist) + 1)
    tr_loss, tr_acc, va_loss, va_acc = hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]
    best_ep = int(np.argmax(va_acc)) + 1
    best_va = va_acc[best_ep - 1]

    # Tag the title with gap-affecting settings so a saved PNG is self-identifying.
    tags = [f"seed {cfg.seed}"]
    if cfg.gradual_unfreeze:
        tags.append(f"gradual-unfreeze/{cfg.unfreeze_every}ep")
    if cfg.dropout > 0:
        tags.append(f"dropout {cfg.dropout}")
    if cfg.use_randaugment:
        tags.append(f"randaug m{cfg.randaugment_magnitude}/n{cfg.randaugment_num_ops}")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{cfg.model_name} fine-tuning — training curves  ({', '.join(tags)})",
                 fontsize=14, fontweight="bold")

    ax_loss.plot(epochs, tr_loss, "o-", label="Train loss", color="tab:blue")
    ax_loss.plot(epochs, va_loss, "s-", label="Val loss", color="tab:orange")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss"); ax_loss.grid(True, alpha=0.3); ax_loss.legend()

    ax_acc.plot(epochs, tr_acc, "o-", label="Train acc", color="tab:blue")
    ax_acc.plot(epochs, va_acc, "s-", label="Val acc", color="tab:orange")
    ax_acc.axhline(BASELINE_ACC, ls="--", color="gray",
                   label=f"Kaggle baseline ({BASELINE_ACC:.0%})")
    ax_acc.scatter([best_ep], [best_va], s=140, facecolors="none",
                   edgecolors="red", linewidths=2, zorder=5)
    ax_acc.annotate(f"best val {best_va:.3f} @ ep {best_ep}",
                    xy=(best_ep, best_va), xytext=(0.55, 0.1),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.02); ax_acc.set_title("Accuracy")
    ax_acc.grid(True, alpha=0.3); ax_acc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(cfg.curves_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    print(f"Saved {cfg.curves_path} | best val {best_va:.3f} @ epoch {best_ep}")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _test_ids(test_dir: Path, scored_ids, predict_all: bool):
    if predict_all:
        return sorted((p.name for p in test_dir.glob("*.jpg")),
                      key=lambda s: int(s.split(".")[0]))
    return sorted(scored_ids, key=lambda s: int(s.split(".")[0]))


def _tta_views(cfg: Config, img: "Image.Image", eval_tfms) -> list[torch.Tensor]:
    """Return the list of normalized tensors to average over for one image.

    Without TTA this is just the single deterministic eval view. With TTA we add
    a horizontal flip and/or a five-crop (4 corners + center). All views share the
    same resize/normalize as eval, so they stay in distribution — TTA only helps
    when the views are this conservative.
    """
    if not cfg.use_tta:
        return [eval_tfms(img)]

    img_size = cfg.spec.img_size
    resize = int(round(img_size * 256 / 224))
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    base_imgs = [img]
    if cfg.tta_hflip:
        base_imgs.append(transforms.functional.hflip(img))

    views: list[torch.Tensor] = []
    for im in base_imgs:
        im = transforms.functional.resize(im, resize)
        if cfg.tta_five_crop:
            for crop in transforms.functional.five_crop(im, img_size):
                views.append(normalize(transforms.functional.to_tensor(crop)))
        else:
            crop = transforms.functional.center_crop(im, img_size)
            views.append(normalize(transforms.functional.to_tensor(crop)))
    return views


@torch.no_grad()
def predict_probs(cfg: Config, model, eval_tfms, *, predict_all=True, return_probs=False):
    """Run the model over the test set.

    Returns a list of (img_id, pred_label). If return_probs=True, also returns
    an (N, NUM_CLASSES) numpy array of softmax probabilities aligned to a
    returned id list — used by the ensemble. When cfg.use_tta is set, softmax
    probabilities are averaged over several augmented views per image.
    """
    test_dir = cfg.data_root / "test"
    sample_sub = cfg.data_root / "sample_submission.csv"
    with open(sample_sub, newline="") as f:
        reader = csv.reader(f); next(reader)
        scored_ids = {row[0] for row in reader}

    ids = _test_ids(test_dir, scored_ids, predict_all)
    model.eval()
    rows, probs = [], []
    for img_id in ids:
        img = Image.open(test_dir / img_id).convert("RGB")
        views = _tta_views(cfg, img, eval_tfms)
        x = torch.stack(views).to(DEVICE)               # (V, C, H, W)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = model(x)
        # Average probabilities across views (not logits — comparably scaled).
        p = torch.softmax(out.float(), dim=1).mean(dim=0).cpu().numpy()
        rows.append((img_id, int(p.argmax())))
        if return_probs:
            probs.append(p)
    if return_probs:
        return ids, np.asarray(probs), scored_ids
    return rows, scored_ids


def write_submission(path, rows, scored_ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        w.writerows(rows)
    predicted = {r[0] for r in rows}
    missing = scored_ids - predicted
    covered = len(scored_ids - missing)
    print(f"Wrote {len(rows)} predictions to {path}")
    print(f"Scored IDs covered: {covered}/{len(scored_ids)}"
          + (f"  MISSING {len(missing)}" if missing else "  (all present)"))
    return missing


def load_checkpoint(cfg: Config, model):
    ckpt = torch.load(cfg.ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {cfg.ckpt_path} (val acc {ckpt['val_acc']:.3f})")
    return ckpt


def load_model_from_checkpoint(model_name: str, *, seed: int = SEED,
                               gradual_unfreeze: bool = False,
                               use_randaugment: bool = False,
                               use_tta=False, tta_five_crop=False):
    """Build a model whose head matches its checkpoint, load it, and return
    ``(cfg, model, ckpt)``. ``seed``, ``gradual_unfreeze`` and ``use_randaugment``
    only select the checkpoint *path* (via _run_suffix) — at eval the trained
    model is a normal fully-unfrozen network regardless of how it was trained
    (RandAugment is train-only). The stored ``dropout`` is read back so a
    dropout-trained model reconstructs the same Sequential head (otherwise the
    state-dict keys would not match). TTA flags are passed through so callers can
    drive inference from the returned cfg.
    """
    loc = dict(seed=seed, gradual_unfreeze=gradual_unfreeze,
               use_randaugment=use_randaugment)
    peek = torch.load(Config(model_name=model_name, **loc).ckpt_path,
                      map_location="cpu", weights_only=False)
    cfg = Config(model_name=model_name, **loc,
                 dropout=float(peek.get("dropout", 0.0)),
                 use_tta=use_tta, tta_five_crop=tta_five_crop)
    model = build_model(cfg)
    ckpt = load_checkpoint(cfg, model)
    return cfg, model, ckpt


def run_inference(cfg: Config, model, eval_tfms, *, predict_all=True):
    rows, scored_ids = predict_probs(cfg, model, eval_tfms, predict_all=predict_all)
    write_submission(cfg.submission_path, rows, scored_ids)
    return rows
