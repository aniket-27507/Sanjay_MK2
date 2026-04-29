#!/usr/bin/env python3
"""Export a trained LidarWorldModel to ONNX with fixed input shapes.

Jetson TensorRT prefers static shapes, so we export with batch=1 and
the polar-grid sizes baked into the graph. The exporter performs a
torch-vs-onnxruntime round-trip and exits non-zero if the maximum
absolute logit divergence exceeds 1e-3.

Usage:

    python scripts/export_lidar_world_model_onnx.py \\
        --ckpt runs/lidar_world_model/best.pt \\
        --out lidar_world_model.onnx
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.single_drone.world_model.lidar_world_model import (  # noqa: E402
    LidarWorldModel,
    LidarWorldModelConfig,
)


def _build_model_from_ckpt(ckpt: Dict[str, Any]) -> LidarWorldModel:
    if "model_config" in ckpt:
        cfg = LidarWorldModelConfig(**ckpt["model_config"])
        return LidarWorldModel(cfg)
    return LidarWorldModel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export LidarWorldModel → ONNX")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--max-abs-diff", type=float, default=1e-3, help="Round-trip tolerance"
    )
    args = parser.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = model.cfg
    dummy_inputs = torch.randn(
        1, cfg.history_frames, cfg.n_input_channels, cfg.n_height_bands, cfg.n_sectors
    )
    dummy_motion = torch.randn(1, cfg.history_frames, cfg.motion_dim)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Force the legacy TorchScript-based exporter so we don't depend on
    # onnxscript at deployment time. Jetson TensorRT prefers static-shape
    # ONNX, which is also what the legacy exporter produces by default.
    torch.onnx.export(
        model,
        (dummy_inputs, dummy_motion),
        str(out_path),
        opset_version=args.opset,
        input_names=["inputs", "motion"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )

    # Round-trip
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        print(f"onnxruntime not available; skipping round-trip ({exc})", file=sys.stderr)
        return 0

    with torch.no_grad():
        torch_out = model(dummy_inputs, dummy_motion).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        ["logits"],
        {
            "inputs": dummy_inputs.numpy(),
            "motion": dummy_motion.numpy(),
        },
    )[0]

    diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"[onnx-export] wrote {out_path} (opset={args.opset}); max_abs_diff={diff:.6g}")
    print(f"[onnx-export] model_config={asdict(cfg)}")

    if diff > float(args.max_abs_diff):
        print(
            f"ERROR: max_abs_diff {diff:.6g} exceeds tolerance {args.max_abs_diff:.6g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
