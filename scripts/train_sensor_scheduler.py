"""
Train the SensorScheduler policy network with PPO (stable-baselines3).

Usage::

    # Fresh run, 200k timesteps
    python scripts/train_sensor_scheduler.py --total-steps 200000

    # Resume from existing checkpoint
    python scripts/train_sensor_scheduler.py --resume runs/sensor_scheduler/policy.zip

    # Colab pattern: resume if a previous run exists, otherwise start fresh
    python scripts/train_sensor_scheduler.py --auto-resume --total-steps 1000000

The trained policy can then be wrapped via RLPolicy in src/single_drone/sensor_scheduler.py
(future Step 5C work) and dropped into the SensorScheduler.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Imports inside main() so that --help works even if SB3 isn't installed.
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="PPO training for SensorScheduler")
    p.add_argument("--total-steps", type=int, default=200_000,
                   help="Total environment timesteps to train for")
    p.add_argument("--episode-duration", type=float, default=60.0,
                   help="Sim seconds per training episode (shorter = more diverse)")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel envs (1 is fine; scenario sim is the bottleneck)")
    p.add_argument("--save-dir", type=Path, default=Path("runs/sensor_scheduler"),
                   help="Where to save the policy zip and TensorBoard logs")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a previous policy.zip to resume from")
    p.add_argument("--auto-resume", action="store_true",
                   help="Resume from <save-dir>/policy.zip if it exists")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tb", action="store_true",
                   help="Enable TensorBoard logging at <save-dir>/tb")
    return p.parse_args()


def build_env(episode_duration: float, seed: int):
    """Construct one SensorSchedulerEnv. Imported lazily so --help works without gym."""
    from src.single_drone.sensor_scheduler_env import SensorSchedulerEnv
    return SensorSchedulerEnv(episode_duration_sec=episode_duration, seed=seed)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Defer SB3 / gym imports so the script's --help works without them.
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        logger.error("stable-baselines3 not installed: %s", e)
        logger.error("install with: pip install stable-baselines3[extra] gymnasium")
        sys.exit(2)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.save_dir / "policy.zip"
    tb_log = str(args.save_dir / "tb") if args.tb else None

    # Resolve resume source
    resume_from: "Path | None" = None
    if args.resume is not None:
        resume_from = args.resume
    elif args.auto_resume and final_path.exists():
        resume_from = final_path

    env_kwargs = dict(episode_duration=args.episode_duration, seed=args.seed)
    vec_env = make_vec_env(
        env_id=lambda: build_env(**env_kwargs),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    if resume_from is not None and resume_from.exists():
        logger.info("Resuming PPO from %s", resume_from)
        model = PPO.load(resume_from, env=vec_env, tensorboard_log=tb_log)
    else:
        logger.info("Starting fresh PPO run (seed=%d)", args.seed)
        # Tiny MLP per docs/ARCHITECTURE.md: ~3,500 params target.
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=dict(net_arch=[64, 32]),
            n_steps=512,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tb_log,
        )

    logger.info("Training for %d timesteps", args.total_steps)
    model.learn(total_timesteps=args.total_steps, reset_num_timesteps=False)
    model.save(final_path)
    logger.info("Saved policy to %s", final_path)


if __name__ == "__main__":
    main()
