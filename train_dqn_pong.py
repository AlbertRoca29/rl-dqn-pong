"""Train a DQN agent on Pong with Stable-Baselines3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from wrappers import DEFAULT_ENV_ID, make_eval_env, make_train_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Pong")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)

    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="dqn_pong")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.log_dir) / args.run_name
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(
        env_id=args.env_id,
        n_envs=1,
        seed=args.seed,
        frame_stack=args.frame_stack,
        clip_reward=True,
    )
    eval_env = make_eval_env(
        env_id=args.env_id,
        seed=args.seed + 123,
        frame_stack=args.frame_stack,
        clip_reward=False,
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
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))

    hparams = {
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "frame_stack": args.frame_stack,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
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
    }
    with (run_dir / "hparams.json").open("w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)

    train_env.close()
    eval_env.close()

    print(f"Training complete. Final model: {final_model_path}.zip")
    print(f"Best eval model (if improved): {model_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
