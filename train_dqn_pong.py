"""Train a DQN agent on Pong with Stable-Baselines3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from wrappers import DEFAULT_ENV_ID, make_eval_env, make_train_env


def get_policy_kwargs(policy_variant: str) -> dict:
    """Return policy kwargs for predefined CnnPolicy architecture variants."""
    variants = {
        "small": {"net_arch": [256]},
        "base": None,
        "large": {"net_arch": [512, 256]},
    }
    if policy_variant not in variants:
        raise ValueError(
            f"Unknown policy variant '{policy_variant}'. "
            f"Available: {sorted(variants)}"
        )
    return variants[policy_variant]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Pong")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument(
        "--minimal-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reduced Pong action space (noop + two movement actions).",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--policy-variant",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="CnnPolicy head architecture variant.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)

    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="dqn_pong")
    parser.add_argument(
        "--save-replay-buffer",
        type=str,
        default=None,
        help="Path to save replay buffer (.pkl).",
    )
    parser.add_argument(
        "--load-replay-buffer",
        type=str,
        default=None,
        help="Path to load replay buffer (.pkl) before training.",
    )
    parser.add_argument(
        "--skip-learning-starts-when-loading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When loading replay buffer, set effective learning_starts=0 so "
            "training starts immediately."
        ),
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help=(
            "Collect experience only (no SGD) for learning_starts steps and "
            "optionally save replay buffer."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.warmup_only and args.load_replay_buffer is not None:
        raise ValueError("--warmup-only cannot be combined with --load-replay-buffer.")

    effective_learning_starts = args.learning_starts
    if args.load_replay_buffer is not None and args.skip_learning_starts_when_loading:
        effective_learning_starts = 0

    run_dir = Path(args.log_dir) / args.run_name
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(
        env_id=args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        frame_stack=args.frame_stack,
        clip_reward=True,
        minimal_actions=args.minimal_actions,
        vec_env_type=args.vec_env,
    )

    eval_env = None
    eval_callback = None
    if not args.disable_eval:
        eval_env = make_eval_env(
            env_id=args.env_id,
            seed=args.seed + 1,  # Use a different seed for evaluation
            frame_stack=args.frame_stack,
            clip_reward=False,
            minimal_actions=args.minimal_actions,
            render_mode=None,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=args.eval_episodes,
        )

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=get_policy_kwargs(args.policy_variant),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=effective_learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    if args.load_replay_buffer is not None:
        replay_path = Path(args.load_replay_buffer)
        if not replay_path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {replay_path}")
        model.load_replay_buffer(str(replay_path))
        print(f"Loaded replay buffer from {replay_path}")
        if args.skip_learning_starts_when_loading:
            print("Using learning_starts=0 for this run (loaded warm-up buffer).")

    learn_steps = args.learning_starts if args.warmup_only else args.total_timesteps
    model.learn(total_timesteps=learn_steps, callback=eval_callback)

    if args.save_replay_buffer is not None:
        replay_out = Path(args.save_replay_buffer)
        replay_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(str(replay_out))
        print(f"Saved replay buffer to {replay_out}")

    if args.warmup_only:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        print(
            "Warm-up complete. No final model/checkpoint exported "
            "because --warmup-only was set."
        )
        return

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))

    hparams = {
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "frame_stack": args.frame_stack,
        "minimal_actions": args.minimal_actions,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "device": args.device,
        "policy_variant": args.policy_variant,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "effective_learning_starts": effective_learning_starts,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "disable_eval": args.disable_eval,
        "save_replay_buffer": args.save_replay_buffer,
        "load_replay_buffer": args.load_replay_buffer,
        "skip_learning_starts_when_loading": args.skip_learning_starts_when_loading,
        "warmup_only": args.warmup_only,
    }
    with (run_dir / "hparams.json").open("w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)

    train_env.close()
    if eval_env is not None:
        eval_env.close()

    print(f"Training complete. Final model: {final_model_path}.zip")
    print(f"Best eval model (if improved): {model_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
