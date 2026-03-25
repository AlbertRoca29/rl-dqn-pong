"""Evaluate a trained DQN on Pong and export best/worst episode GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN

from wrappers import DEFAULT_ENV_ID, make_eval_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN on Pong")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def run_episode(model: DQN, env, deterministic: bool = True):
    obs = env.reset()
    done = np.array([False])
    episode_reward = 0.0
    frames = []

    while not bool(done[0]):
        # For vectorized env with a single instance, render gives one RGB frame.
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, done, infos = env.step(action)
        episode_reward += float(rewards[0])

    return episode_reward, frames


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(
        env_id=args.env_id,
        seed=args.seed,
        frame_stack=args.frame_stack,
        clip_reward=False,
        render_mode="rgb_array",
    )

    model = DQN.load(args.model_path, env=env)

    rewards = []
    episodes = []
    for ep in range(args.episodes):
        reward, frames = run_episode(model, env, deterministic=True)
        rewards.append(reward)
        episodes.append({"episode": ep, "reward": reward, "frames": frames})
        print(f"Episode {ep + 1}/{args.episodes} reward: {reward:.2f}")

    rewards_np = np.asarray(rewards, dtype=np.float32)
    mean_reward = float(np.mean(rewards_np))
    std_reward = float(np.std(rewards_np))

    min_idx = int(np.argmin(rewards_np))
    max_idx = int(np.argmax(rewards_np))

    worst = episodes[min_idx]
    best = episodes[max_idx]

    worst_gif = output_dir / "worst_episode.gif"
    best_gif = output_dir / "best_episode.gif"

    if worst["frames"]:
        imageio.mimsave(worst_gif, worst["frames"], fps=args.fps)
    if best["frames"]:
        imageio.mimsave(best_gif, best["frames"], fps=args.fps)

    summary = {
        "episodes": args.episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "worst_episode_index": worst["episode"],
        "worst_reward": float(worst["reward"]),
        "best_episode_index": best["episode"],
        "best_reward": float(best["reward"]),
        "worst_gif": str(worst_gif),
        "best_gif": str(best_gif),
    }

    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation complete")
    print(f"Average reward over {args.episodes} episodes: {mean_reward:.3f} +- {std_reward:.3f}")
    print(f"Worst episode: {worst['episode']} reward={worst['reward']:.2f}")
    print(f"Best episode: {best['episode']} reward={best['reward']:.2f}")
    print(f"Summary JSON: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
