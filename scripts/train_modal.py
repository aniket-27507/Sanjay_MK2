"""
Train the SensorScheduler policy on Modal (modal.com) instead of Colab.

Why Modal:
  - No Jupyter event loop -> no nest_asyncio workaround needed
  - No streaming-output truncation -> full logs preserved
  - Persistent volume for policy.zip survives across runs
  - One-line CLI invocation; auto-downloads weights when done

Prerequisites (one-time, see README of this script for full steps):
  1. Sign up at https://modal.com (free, ~2 min)
  2. pip install modal
  3. modal token new
  4. From the repo root: modal run scripts/train_modal.py

That's it.

Usage::

    # Default fast-mode 300k training, T4 GPU, ~15-25 min
    modal run scripts/train_modal.py

    # Bigger run
    modal run scripts/train_modal.py --total-steps 1000000

    # Eval the trained policy (after training)
    modal run scripts/train_modal.py::eval_policy

    # Inspect the persistent volume contents
    modal volume ls sanjay-mk2-models
    modal volume get sanjay-mk2-models /policy.zip ./policy.zip

@author: Archishman Paul
"""

from __future__ import annotations

import modal


# ════════════════════════════════════════════════════════════════════
#  App, image, volume
# ════════════════════════════════════════════════════════════════════

app = modal.App("sanjay-mk2-rl")

# Container image: thin Debian + Python + RL deps + repo cloned at /workspace.
# Repo clone is baked into the image at build time; runtime does git pull
# so a fresh push doesn't require image rebuild.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "stable-baselines3[extra]==2.3.2",
        "gymnasium==0.29.1",
        "numpy",
        "pyyaml",
        "websockets",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/aniket-27507/Sanjay_MK2.git /workspace/repo",
    )
)

# Persistent volume so policy.zip survives between runs (resume-from-checkpoint).
volume = modal.Volume.from_name("sanjay-mk2-models", create_if_missing=True)


# ════════════════════════════════════════════════════════════════════
#  Train
# ════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,                          # 1 hr max
    volumes={"/models": volume},
)
def train(
    total_steps: int = 1_000_000,
    episode_steps: int = 120,
    n_envs: int = 8,
    seed: int = 42,
) -> bytes:
    """Run PPO training inside Modal's container. Returns policy.zip bytes."""
    import os
    import shutil
    import subprocess

    os.chdir("/workspace/repo")

    # Pull latest in case the image was built before recent pushes
    subprocess.run(["git", "pull", "origin", "main"], check=True)

    save_dir = "/workspace/repo/runs/sensor_scheduler"
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "policy.zip")

    # Resume from previous run if a policy.zip exists in the volume
    volume_ckpt = "/models/policy.zip"
    if os.path.exists(volume_ckpt):
        shutil.copy(volume_ckpt, final_path)
        print(f"[modal] Restored checkpoint from volume: {volume_ckpt}")

    cmd = [
        "python", "scripts/train_sensor_scheduler.py",
        "--env", "fast",
        "--total-steps", str(total_steps),
        "--episode-steps", str(episode_steps),
        "--n-envs", str(n_envs),
        "--seed", str(seed),
        "--auto-resume",
    ]
    print(f"[modal] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Persist back to volume so the next run can resume
    shutil.copy(final_path, volume_ckpt)
    volume.commit()
    print(f"[modal] Saved policy to volume: {volume_ckpt}")

    with open(final_path, "rb") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════════
#  Eval (post-training sanity check)
# ════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/models": volume},
)
def eval_policy(num_seeds: int = 3) -> str:
    """Run heuristic-vs-trained eval episodes on Modal. Returns formatted text.

    Modal's CLI parser doesn't accept ``list[int]`` annotations, so we take
    an integer count and run seeds 0..num_seeds-1.
    """
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/workspace/repo")
    subprocess.run(["git", "pull", "origin", "main"], check=True)

    save_dir = "/workspace/repo/runs/sensor_scheduler"
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "policy.zip")

    if not os.path.exists("/models/policy.zip"):
        return "ERROR: no policy.zip in volume; run `modal run scripts/train_modal.py` first"
    shutil.copy("/models/policy.zip", final_path)

    sys.path.insert(0, "/workspace/repo")
    from stable_baselines3 import PPO
    from src.single_drone.sensor_scheduler_fast_env import SensorSchedulerFastEnv
    from src.single_drone.sensor_scheduler_rl import encode_action

    model = PPO.load(final_path)

    def run_episode(use_trained: bool, seed: int):
        env = SensorSchedulerFastEnv(seed=seed)
        obs, _ = env.reset()
        total_r = 0.0
        n_steps = 0
        rgb_fires = 0
        thermal_fires = 0
        while True:
            if use_trained:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = encode_action(15, 0)   # day-patrol heuristic default
            obs, r, terminated, truncated, info = env.step(int(action))
            total_r += r
            n_steps += 1
            if info["rgb_fps"] > 0:
                rgb_fires += 1
            if info["thermal_fps"] > 0:
                thermal_fires += 1
            if terminated or truncated:
                break
        return dict(reward=round(total_r, 3), steps=n_steps,
                    rgb_active=rgb_fires, thermal_active=thermal_fires)

    lines = []
    print("\n========== EVAL RESULTS ==========")
    for seed in range(num_seeds):
        a = run_episode(use_trained=False, seed=seed)
        b = run_episode(use_trained=True, seed=seed)
        line = f"seed={seed}  heuristic={a}  trained={b}"
        print(line)
        lines.append(line)
    print("==================================\n")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  Local entrypoint -- runs when you call `modal run scripts/train_modal.py`
# ════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main(
    total_steps: int = 1_000_000,
    episode_steps: int = 120,
    n_envs: int = 8,
    seed: int = 42,
    download_to: str = "policy.zip",
):
    """Train, then save the resulting policy.zip to the local filesystem."""
    print(f"[modal] Starting training: total_steps={total_steps} n_envs={n_envs}")
    policy_bytes = train.remote(
        total_steps=total_steps,
        episode_steps=episode_steps,
        n_envs=n_envs,
        seed=seed,
    )
    with open(download_to, "wb") as f:
        f.write(policy_bytes)
    print(f"\n[modal] Saved {len(policy_bytes):,} bytes to {download_to}")
    print(f"[modal] Run eval with:  modal run scripts/train_modal.py::eval_policy")
