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
def eval_policy(num_seeds: int = 10) -> str:
    """Run heuristic-vs-trained eval episodes on Modal. Returns formatted text.

    Compares the trained PPO policy against the *real* HeuristicPolicy
    (state-machine: EMERGENCY_BURST > INSPECT_DUAL > NIGHT_PATROL > DAY_PATROL),
    not a constant action.  Reports per-component reward decomposition
    (detection / compute / switch) plus a day-vs-night condition split,
    so collapse modes like always-on thermal or idle-policy floors are
    visible in one run instead of needing a follow-up forensic pass.

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
    from src.single_drone.sensor_scheduler import (
        FPS_OFF, HardRails, HeuristicPolicy, NIGHT_LUX_THRESHOLD,
    )
    from src.single_drone.sensor_scheduler_fast_env import SensorSchedulerFastEnv
    from src.single_drone.sensor_scheduler_rl import (
        compute_reward_breakdown, encode_action,
    )

    model = PPO.load(final_path)
    heuristic = HeuristicPolicy()

    def _heuristic_action_idx(env: SensorSchedulerFastEnv) -> int:
        """Real HeuristicPolicy decision, encoded back into the discrete
        action space. Rails are applied inside env.step() exactly as for
        the trained policy, so the comparison is apples-to-apples."""
        state = env.current_sensor_state()
        action = heuristic.decide(state)
        return encode_action(action.rgb_fps, action.thermal_fps)

    def run_episode(use_trained: bool, seed: int):
        env = SensorSchedulerFastEnv(seed=seed)
        obs, info0 = env.reset()
        total_det = 0.0
        total_compute = 0.0
        total_switch = 0.0
        total_r = 0.0
        n_steps = 0
        rgb_fires = 0
        thermal_fires = 0
        # Per-condition split: night = lux < NIGHT_LUX_THRESHOLD (rail R1 zone)
        night_steps = 0
        night_thermal_fires = 0
        day_steps = 0
        day_thermal_fires = 0
        prev_rgb = FPS_OFF
        prev_thermal = FPS_OFF
        first_step = True
        while True:
            if use_trained:
                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(action_arr)
            else:
                action = _heuristic_action_idx(env)
            obs, r, terminated, truncated, info = env.step(action)

            # Reward decomposition uses POST-RAILS fps (info dict carries those).
            # We don't have the per-step detection list here, but the breakdown
            # of compute + switch is exact; detection_reward is recoverable as
            # (total - compute_penalty - switch_penalty).
            bd = compute_reward_breakdown(
                detected_objects=[],   # detections aren't exposed; back out below
                rgb_fps=info["rgb_fps"],
                thermal_fps=info["thermal_fps"],
                prev_rgb_fps=None if first_step else prev_rgb,
                prev_thermal_fps=None if first_step else prev_thermal,
            )
            compute_pen = bd.compute_penalty
            switch_pen = bd.switch_penalty
            det_reward = r + compute_pen + switch_pen   # invert reward formula

            total_det += det_reward
            total_compute += compute_pen
            total_switch += switch_pen
            total_r += r
            n_steps += 1
            if info["rgb_fps"] > 0:
                rgb_fires += 1
            if info["thermal_fps"] > 0:
                thermal_fires += 1
            if info["lux"] < NIGHT_LUX_THRESHOLD:
                night_steps += 1
                if info["thermal_fps"] > 0:
                    night_thermal_fires += 1
            else:
                day_steps += 1
                if info["thermal_fps"] > 0:
                    day_thermal_fires += 1

            prev_rgb = info["rgb_fps"]
            prev_thermal = info["thermal_fps"]
            first_step = False
            if terminated or truncated:
                break

        return dict(
            reward=round(total_r, 3),
            det=round(total_det, 3),
            compute=round(total_compute, 3),
            switch=round(total_switch, 3),
            steps=n_steps,
            rgb_active=rgb_fires,
            thermal_active=thermal_fires,
            night_steps=night_steps,
            night_thermal=night_thermal_fires,
            day_steps=day_steps,
            day_thermal=day_thermal_fires,
            init_lux=round(info0["lux"], 1),
            init_mission=info0["mission_state"],
        )

    def _fmt_row(label: str, e: dict) -> str:
        # Compact one-line summary: total | components | sensor on-rate | day/night thermal
        thermal_day = f"{e['day_thermal']}/{e['day_steps']}" if e['day_steps'] else "-"
        thermal_night = f"{e['night_thermal']}/{e['night_steps']}" if e['night_steps'] else "-"
        return (
            f"  {label:9s} R={e['reward']:8.2f}  det={e['det']:7.2f}  "
            f"compute={e['compute']:6.2f}  switch={e['switch']:5.2f}  "
            f"rgb={e['rgb_active']}/{e['steps']}  thermal={e['thermal_active']}/{e['steps']}  "
            f"th_day={thermal_day}  th_night={thermal_night}"
        )

    lines = []
    print("\n" + "=" * 110)
    print(f" EVAL: trained vs real HeuristicPolicy  ({num_seeds} seeds)")
    print(" Reward = detection - compute_penalty - switch_penalty")
    print(" Goal: trained >= heuristic across seeds; lower compute when no detections to gain")
    print("=" * 110)
    wins = 0
    for seed in range(num_seeds):
        h = run_episode(use_trained=False, seed=seed)
        t = run_episode(use_trained=True, seed=seed)
        delta = t["reward"] - h["reward"]
        winner = "trained" if delta > 0 else "heuristic"
        if delta > 0:
            wins += 1
        header = (
            f"seed={seed}  init_lux={h['init_lux']}  init_mission={h['init_mission']}  "
            f"delta={delta:+.2f} (winner={winner})"
        )
        print(header)
        h_row = _fmt_row("heuristic", h)
        t_row = _fmt_row("trained",   t)
        print(h_row)
        print(t_row)
        lines.extend([header, h_row, t_row, ""])
    print("=" * 110)
    print(f" trained wins {wins}/{num_seeds} seeds")
    print("=" * 110 + "\n")
    lines.append(f"trained wins {wins}/{num_seeds} seeds")
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
