"""Train the LiDAR predictive-occupancy world model on Modal.

Mirrors ``scripts/train_modal.py`` (which trains the SensorScheduler RL
policy). Reuses the existing ``sanjay-mk2-rl`` Modal app and the
``sanjay-mk2-models`` checkpoint volume; introduces a separate
``sanjay-mk2-lidar-data`` volume for shard storage.

One-time setup:

    pip install modal && modal token new

Upload shards from your workstation (one-time, after running
``scripts/build_lidar_world_dataset.py`` locally):

    modal volume put sanjay-mk2-lidar-data ./data/lidar_world_model /lidar_world_model

Train:

    modal run scripts/train_lidar_world_model_modal.py
    modal run scripts/train_lidar_world_model_modal.py --epochs 60 --batch-size 256

The trained ``best.pt`` is downloaded back to ``./best.pt`` on completion.
"""

from __future__ import annotations

import modal


# ════════════════════════════════════════════════════════════════════
#  App, image, volumes
# ════════════════════════════════════════════════════════════════════

app = modal.App("sanjay-mk2-rl")  # reuse the existing app

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "numpy",
        "pyyaml",
        "onnx",
        "onnxruntime",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/aniket-27507/Sanjay_MK2.git /workspace/repo",
    )
)

models_volume = modal.Volume.from_name("sanjay-mk2-models", create_if_missing=True)
data_volume = modal.Volume.from_name("sanjay-mk2-lidar-data", create_if_missing=True)


# ════════════════════════════════════════════════════════════════════
#  Train
# ════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/models": models_volume, "/data": data_volume},
)
def train(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
) -> bytes:
    """Run training inside Modal's container. Returns ``best.pt`` bytes."""
    import os
    import shutil
    import subprocess

    os.chdir("/workspace/repo")
    subprocess.run(["git", "pull", "origin", "main"], check=True)

    save_dir_local = "/workspace/repo/runs/lidar_world_model"
    os.makedirs(save_dir_local, exist_ok=True)
    last_local = os.path.join(save_dir_local, "last.pt")
    best_local = os.path.join(save_dir_local, "best.pt")

    # Restore previous checkpoint if present
    last_volume = "/models/lidar_world_model/last.pt"
    if os.path.exists(last_volume):
        shutil.copy(last_volume, last_local)
        print(f"[modal] Restored checkpoint from volume: {last_volume}")

    cmd = [
        "python",
        "scripts/train_lidar_world_model.py",
        "--data",
        "/data/lidar_world_model",
        "--save-dir",
        save_dir_local,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--auto-resume",
    ]
    print(f"[modal] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Persist back to volume
    os.makedirs("/models/lidar_world_model", exist_ok=True)
    if os.path.exists(best_local):
        shutil.copy(best_local, "/models/lidar_world_model/best.pt")
    if os.path.exists(last_local):
        shutil.copy(last_local, "/models/lidar_world_model/last.pt")
    models_volume.commit()
    print("[modal] Saved best.pt and last.pt to volume sanjay-mk2-models")

    if os.path.exists(best_local):
        with open(best_local, "rb") as f:
            return f.read()
    raise RuntimeError("Training completed but no best.pt was produced")


# ════════════════════════════════════════════════════════════════════
#  Local entrypoint
# ════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
    download_to: str = "best.pt",
):
    """Train, then save the resulting best.pt to the local filesystem."""
    print(f"[modal] Training: epochs={epochs} batch_size={batch_size} lr={lr}")
    weights = train.remote(
        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed
    )
    with open(download_to, "wb") as f:
        f.write(weights)
    print(f"\n[modal] Saved {len(weights):,} bytes to {download_to}")
