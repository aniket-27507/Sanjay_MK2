"""Tests for ``LidarWorldModel`` — forward shape, parameter budget,
gradient flow, and basic invariance to S-axis rolls (sanity check that
the circular-padded conv path is wired)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.single_drone.world_model.lidar_dataset_io import ShardWriter, WindowSample
from src.single_drone.world_model.lidar_world_model import (
    LidarWorldModel,
    LidarWorldModelConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_forward_shape_default_config():
    model = LidarWorldModel()
    inputs = torch.randn(2, 10, 4, 6, 72)
    motion = torch.randn(2, 10, 5)
    out = model(inputs, motion)
    assert out.shape == (2, 4, 6, 72)
    assert out.dtype == torch.float32


def test_forward_shape_custom_horizons():
    cfg = LidarWorldModelConfig(n_horizons=2)
    model = LidarWorldModel(cfg)
    inputs = torch.randn(1, 10, 4, 6, 72)
    motion = torch.randn(1, 10, 5)
    out = model(inputs, motion)
    assert out.shape == (1, 2, 6, 72)


def test_param_count_under_budget():
    model = LidarWorldModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 120_000, f"Model has {n_params} params; budget is <120k"
    assert n_params > 30_000, f"Model has {n_params} params; suspiciously small"


def test_input_shape_validation():
    model = LidarWorldModel()
    inputs = torch.randn(1, 7, 4, 6, 72)  # wrong T
    motion = torch.randn(1, 7, 5)
    try:
        model(inputs, motion)
    except ValueError as exc:
        assert "shape mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for mismatched input shape")


def test_gradient_flows_to_inputs():
    model = LidarWorldModel()
    inputs = torch.randn(1, 10, 4, 6, 72, requires_grad=True)
    motion = torch.randn(1, 10, 5, requires_grad=True)
    out = model(inputs, motion)
    loss = out.sum()
    loss.backward()
    assert inputs.grad is not None
    assert motion.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(motion.grad).all()


def test_circular_pad_used_via_s_axis_translation_equivariance():
    """If the encoder used zero-padded convs the network would treat sector 71
    and sector 0 as separated, so a yaw rotation (S-axis roll) of the input
    would *not* produce the same roll on the output near the wrap. With
    circular padding it should — within numerical tolerance, ignoring the
    motion-conditioned FiLM (zero motion is rotation-invariant)."""
    torch.manual_seed(0)
    model = LidarWorldModel()
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(1, 10, 4, 6, 72)
        motion = torch.zeros(1, 10, 5)
        out = model(inputs, motion)

        rolled_inputs = torch.roll(inputs, shifts=10, dims=-1)
        rolled_out = model(rolled_inputs, motion)
        expected = torch.roll(out, shifts=10, dims=-1)

        # Allow some slack — GroupNorm interacts with global statistics so
        # exact equivariance only holds under per-sample normalisation. The
        # roll relationship should still hold within ~1e-3.
        max_diff = (rolled_out - expected).abs().max().item()
        assert max_diff < 1e-3, f"S-axis roll equivariance broke: max_diff={max_diff}"


def _write_synthetic_dataset(root: Path, n_train: int = 8, n_val: int = 4) -> None:
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        writer = ShardWriter(d, max_windows_per_shard=max(1, n))
        for i in range(n):
            writer.append(
                WindowSample(
                    inputs=rng.standard_normal((10, 4, 6, 72)).astype(np.float32),
                    motion=rng.standard_normal((10, 5)).astype(np.float32),
                    targets=(rng.uniform(0.0, 1.0, size=(4, 6, 72)) > 0.95).astype(np.uint8),
                    timestamp=float(i),
                    source_id=0,
                    pose_compensated=True,
                )
            )
        writer.finalize(target_horizons_s=[0.5, 1.0, 1.5, 2.0])


def test_train_script_smoke(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_synthetic_dataset(data_root, n_train=8, n_val=4)
    save_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lidar_world_model.py"),
            "--data",
            str(data_root),
            "--save-dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--smoke",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"train_lidar_world_model.py failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "best.pt").exists()
    metrics = json.loads((save_dir / "metrics.json").read_text())
    assert isinstance(metrics, list) and len(metrics) >= 1
    assert "train_loss" in metrics[0]
    assert "val_loss" in metrics[0]


def test_train_script_resume(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_synthetic_dataset(data_root, n_train=4, n_val=2)
    save_dir = tmp_path / "runs"

    # First run — produces last.pt at epoch 0
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lidar_world_model.py"),
            "--data",
            str(data_root),
            "--save-dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--smoke",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    # Second run with --auto-resume should not error, and should be a no-op
    # (start_epoch == epochs).
    result2 = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lidar_world_model.py"),
            "--data",
            str(data_root),
            "--save-dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--smoke",
            "--auto-resume",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result2.returncode == 0, result2.stderr
    assert "resumed from" in result2.stdout


def test_eval_script_writes_report(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_synthetic_dataset(data_root, n_train=8, n_val=4)
    save_dir = tmp_path / "runs"

    # Train briefly to produce a real checkpoint
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lidar_world_model.py"),
            "--data",
            str(data_root),
            "--save-dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--smoke",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )

    # Use the val split as the test split for the eval smoke
    report_path = tmp_path / "eval.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_lidar_world_model.py"),
            "--ckpt",
            str(save_dir / "best.pt"),
            "--data",
            str(data_root / "val"),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--batch-size",
            "4",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"eval_lidar_world_model.py failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    report = json.loads(report_path.read_text())
    for key in (
        "f1_per_horizon",
        "iou_per_horizon",
        "ece_per_horizon",
        "front_recall_per_horizon",
        "side_recall_per_horizon",
        "rear_recall_per_horizon",
        "fn_in_tube_per_horizon",
    ):
        assert key in report["metrics"]
        assert len(report["metrics"][key]) == 4


def test_onnx_export_round_trip(tmp_path: Path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    data_root = tmp_path / "data"
    _write_synthetic_dataset(data_root, n_train=4, n_val=2)
    save_dir = tmp_path / "runs"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_lidar_world_model.py"),
            "--data",
            str(data_root),
            "--save-dir",
            str(save_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--smoke",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )

    out = tmp_path / "lidar_world_model.onnx"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_lidar_world_model_onnx.py"),
            "--ckpt",
            str(save_dir / "best.pt"),
            "--out",
            str(out),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ONNX export failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert out.exists()
    assert "max_abs_diff" in result.stdout

    # Independent re-check via onnxruntime
    import onnxruntime as ort  # type: ignore

    from src.single_drone.world_model.lidar_world_model import (
        LidarWorldModel,
        LidarWorldModelConfig,
    )

    ckpt = torch.load(save_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = LidarWorldModelConfig(**ckpt["model_config"])
    model = LidarWorldModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    inputs = torch.randn(1, cfg.history_frames, cfg.n_input_channels,
                         cfg.n_height_bands, cfg.n_sectors)
    motion = torch.randn(1, cfg.history_frames, cfg.motion_dim)

    with torch.no_grad():
        torch_out = model(inputs, motion).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"inputs": inputs.numpy(), "motion": motion.numpy()})[0]
    assert np.max(np.abs(torch_out - onnx_out)) < 1e-3
