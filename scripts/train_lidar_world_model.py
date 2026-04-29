#!/usr/bin/env python3
"""Train the LiDAR predictive-occupancy world model on shards.

Reads ``config/training/lidar_world_model.yaml`` for grid + temporal +
model + training + loss parameters, and ``data/lidar_world_model/`` for
shards produced by ``scripts/build_lidar_world_dataset.py``.

Outputs:
    runs/lidar_world_model/best.pt      best val-loss checkpoint
    runs/lidar_world_model/last.pt      most recent epoch checkpoint
    runs/lidar_world_model/metrics.json per-epoch train/val loss + val F1

Smoke run (1 epoch, a few batches, no DataLoader workers):

    python scripts/train_lidar_world_model.py \\
        --data data/lidar_world_model \\
        --epochs 2 --batch-size 8 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.single_drone.world_model.lidar_world_model import (  # noqa: E402
    LidarWorldModel,
    LidarWorldModelConfig,
)
from src.single_drone.world_model.losses import (  # noqa: E402
    LidarWorldModelLoss,
    compute_pos_weight_per_band,
)
from src.single_drone.world_model.torch_dataset import LidarWorldShardDataset  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_model(cfg_yaml: Dict[str, Any]) -> LidarWorldModel:
    grid = cfg_yaml.get("grid", {})
    temporal = cfg_yaml.get("temporal", {})
    model = cfg_yaml.get("model", {})
    return LidarWorldModel(
        LidarWorldModelConfig(
            history_frames=int(temporal.get("history_frames", 10)),
            n_input_channels=len(grid.get("channels", ["min_range", "occupancy", "point_count", "mean_range"])),
            n_height_bands=int(grid.get("n_height_bands", 6)),
            n_sectors=int(grid.get("n_sectors", 72)),
            n_horizons=len(temporal.get("future_horizons_s", [0.5, 1.0, 1.5, 2.0])),
            channels_stem=int(model.get("channels_stem", 32)),
            channels_block1=int(model.get("channels_block1", 48)),
            channels_block2=int(model.get("channels_block2", 48)),
            motion_film_hidden=int(model.get("motion_film_hidden", 32)),
        )
    )


def _f1_at_threshold(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-horizon F1; returns shape (F,)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * targets).sum(dim=(0, 2, 3))
    fp = (pred * (1.0 - targets)).sum(dim=(0, 2, 3))
    fn = ((1.0 - pred) * targets).sum(dim=(0, 2, 3))
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def _epoch_loop(
    model: LidarWorldModel,
    loss_fn: LidarWorldModelLoss,
    loader: DataLoader,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    smoke_max_batches: int | None = None,
) -> Dict[str, float]:
    is_train = optim is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    f1_sum = None
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if smoke_max_batches is not None and batch_idx >= smoke_max_batches:
            break
        inputs = batch["inputs"].to(device)
        motion = batch["motion"].to(device)
        targets = batch["targets"].to(device)
        if is_train:
            optim.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(inputs, motion)
            loss = loss_fn(logits, targets)
        if is_train:
            loss.backward()
            optim.step()
        bs = inputs.shape[0]
        total_loss += float(loss.item()) * bs
        total_samples += bs
        n_batches += 1
        with torch.no_grad():
            batch_f1 = _f1_at_threshold(logits.detach(), targets)
            f1_sum = batch_f1 if f1_sum is None else f1_sum + batch_f1

    if total_samples == 0:
        return {"loss": float("nan"), "f1_per_horizon": []}
    f1_avg = (f1_sum / max(n_batches, 1)).cpu().tolist() if f1_sum is not None else []
    return {"loss": total_loss / total_samples, "f1_per_horizon": f1_avg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LiDAR predictive-occupancy world model")
    parser.add_argument("--data", required=True, help="Root with train/, val/, test/ shard subdirs")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "training" / "lidar_world_model.yaml"),
        help="Training YAML (defaults to repo's config/training/lidar_world_model.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-dir", default=str(PROJECT_ROOT / "runs" / "lidar_world_model"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny run: 1 epoch, 4 batches each split")
    parser.add_argument("--auto-resume", action="store_true", help="Load last.pt if present")
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Skip the per-band pos_weight prepass (use clip[1] for all bands)",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    training = cfg.get("training", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    grid = cfg.get("grid", {}) or {}
    temporal = cfg.get("temporal", {}) or {}

    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 30))
    batch_size = int(args.batch_size if args.batch_size is not None else training.get("batch_size", 64))
    lr = float(args.lr if args.lr is not None else training.get("lr", 3e-4))
    lr_min = float(training.get("lr_min", 1e-5))
    weight_decay = float(training.get("weight_decay", 1e-4))
    motion_noise_std = float(training.get("motion_noise_std", 0.05))
    num_workers = int(
        args.num_workers if args.num_workers is not None else training.get("num_workers", 4)
    )
    seed = int(args.seed if args.seed is not None else training.get("seed", 42))
    if args.smoke:
        epochs = min(epochs, 1)
        num_workers = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(args.device)

    data_root = Path(args.data)
    train_dir = data_root / cfg.get("train", "train")
    val_dir = data_root / cfg.get("val", "val")

    train_ds = LidarWorldShardDataset(
        train_dir, augment=True, motion_noise_std=motion_noise_std, seed=seed
    )
    val_ds = LidarWorldShardDataset(val_dir, augment=False) if val_dir.exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
        )
        if val_ds is not None
        else None
    )

    # Pos weight prepass.
    n_bands = int(grid.get("n_height_bands", 6))
    pos_weight_clip = list(loss_cfg.get("pos_weight_clip", [3.0, 10.0]))
    if args.no_pos_weight or args.smoke:
        pos_weight_per_band = [pos_weight_clip[1]] * n_bands
    else:
        pos_weight_per_band = compute_pos_weight_per_band(
            train_dir, n_height_bands=n_bands, clip=pos_weight_clip
        )
    print(f"[train] pos_weight_per_band = {pos_weight_per_band}")

    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model has {n_params:,} parameters; device={device}")

    loss_fn = LidarWorldModelLoss(
        n_horizons=len(temporal.get("future_horizons_s", [0.5, 1.0, 1.5, 2.0])),
        n_height_bands=n_bands,
        n_sectors=int(grid.get("n_sectors", 72)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        horizon_weights=list(loss_cfg.get("horizon_weights", [1.0, 0.8, 0.6, 0.4])),
        sector_front_bias_k=float(loss_cfg.get("sector_front_bias_k", 2.0)),
        band_distance_tau=float(loss_cfg.get("band_distance_tau", 3.0)),
        pos_weight_clip=pos_weight_clip,
        pos_weight_per_band=pos_weight_per_band,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(epochs, 1), eta_min=lr_min
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"
    metrics_path = save_dir / "metrics.json"

    start_epoch = 0
    best_val = float("inf")
    history: list[Dict[str, Any]] = []
    if args.auto_resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        history = list(ckpt.get("history", []))
        print(f"[train] resumed from {last_path} at epoch {start_epoch}")

    smoke_batches = 4 if args.smoke else None

    for epoch in range(start_epoch, epochs):
        train_metrics = _epoch_loop(
            model, loss_fn, train_loader, optim, device, smoke_max_batches=smoke_batches
        )
        val_metrics = (
            _epoch_loop(
                model, loss_fn, val_loader, None, device, smoke_max_batches=smoke_batches
            )
            if val_loader is not None
            else {"loss": float("nan"), "f1_per_horizon": []}
        )
        scheduler.step()

        line = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["f1_per_horizon"]:
            f1s = ",".join(f"{v:.3f}" for v in val_metrics["f1_per_horizon"])
            line += f" val_f1=[{f1s}]"
        print(line)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_f1_per_horizon": val_metrics["f1_per_horizon"],
                "lr": float(optim.param_groups[0]["lr"]),
            }
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": asdict(model.cfg),
            "best_val": best_val,
            "history": history,
        }
        torch.save(ckpt, last_path)

        if val_metrics["loss"] < best_val and val_loader is not None:
            best_val = float(val_metrics["loss"])
            torch.save(ckpt, best_path)
            print(f"[train]  ↳ new best_val={best_val:.4f}; wrote {best_path}")

    if not best_path.exists():
        # No val loop ran (or only val NaN) — copy last.pt as best so eval works.
        torch.save(torch.load(last_path, map_location="cpu", weights_only=False), best_path)
    metrics_path.write_text(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
