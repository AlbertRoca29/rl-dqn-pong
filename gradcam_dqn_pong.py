"""Extract Grad-CAM maps from a trained DQN Pong agent.

Also exports an x-axis (column-wise) importance histogram by aggregating
Grad-CAM energy along the vertical axis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import DQN

from wrappers import DEFAULT_ENV_ID, make_eval_env


class GradCamExtractor:
    """Simple Grad-CAM helper for a chosen convolutional layer."""

    def __init__(self, conv_layer: torch.nn.Module):
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_handle = conv_layer.register_forward_hook(self._on_forward)
        self._backward_handle = conv_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module, _inputs, output):
        self._activations = output

    def _on_backward(self, _module, _grad_inputs, grad_outputs):
        self._gradients = grad_outputs[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def compute_cam(self) -> np.ndarray:
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM buffers are empty; forward/backward was not run.")

        acts = self._activations
        grads = self._gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM analysis for DQN Pong")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument(
        "--minimal-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reduced Pong action space (must match training setup).",
    )
    parser.add_argument("--output-dir", type=str, default="gradcam_analysis")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Compute Grad-CAM every N environment steps (>=1).",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=20_000,
        help="Safety cap to avoid endless episodes.",
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Grad-CAM overlays for sampled frames.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic action selection during rollout.",
    )
    parser.add_argument(
        "--save-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export one Grad-CAM GIF per evaluated episode.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frames per second for exported Grad-CAM GIFs.",
    )
    return parser.parse_args()


def find_last_conv_layer(model: DQN) -> torch.nn.Module:
    """Find the last Conv2d layer in the online Q-network feature extractor."""
    conv_layers = [
        m for m in model.policy.q_net.features_extractor.modules() if isinstance(m, torch.nn.Conv2d)
    ]
    if not conv_layers:
        raise RuntimeError("No Conv2d layers found in DQN feature extractor.")
    return conv_layers[-1]


def overlay_heatmap(grayscale_frame: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Create an RGB overlay by blending grayscale input and Grad-CAM heatmap."""
    frame_u8 = np.clip(grayscale_frame, 0, 255).astype(np.uint8)
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(gray_bgr, 0.55, heatmap, 0.45, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def column_importance_from_cam(cam: np.ndarray) -> np.ndarray:
    """Collapse CAM along y-axis and normalize into a probability-like histogram."""
    col = cam.sum(axis=0)
    total = float(col.sum())
    if total <= 0.0:
        return np.full_like(col, fill_value=1.0 / max(len(col), 1), dtype=np.float32)
    return (col / total).astype(np.float32)


def save_histogram_image(hist: np.ndarray, out_path: Path) -> None:
    """Save a simple histogram image without adding extra plotting dependencies."""
    width = int(hist.shape[0])
    height = 240
    margin_top = 12
    margin_bottom = 24

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_h = height - margin_top - margin_bottom
    max_val = float(np.max(hist))
    if max_val <= 0.0:
        max_val = 1.0

    norm = hist / max_val
    bar_h = np.clip((norm * plot_h).astype(np.int32), 0, plot_h)

    for x in range(width):
        y0 = height - margin_bottom
        y1 = y0 - int(bar_h[x])
        canvas[y1:y0, x, :] = (60, 120, 220)

    # Axis baseline
    canvas[height - margin_bottom : height - margin_bottom + 1, :, :] = (30, 30, 30)
    cv2.imwrite(str(out_path), canvas)


def extract_latest_grayscale(obs: np.ndarray) -> np.ndarray:
    """Extract latest stacked grayscale frame from vectorized observation."""
    if obs.ndim != 4:
        raise ValueError(f"Expected obs with 4 dims (N + image), got shape {obs.shape}")

    # Channel-last stack: (N, H, W, C)
    if obs.shape[-1] > 1 and obs.shape[1] > 16 and obs.shape[2] > 16:
        return obs[0, :, :, -1]

    # Channel-first stack: (N, C, H, W)
    if obs.shape[1] > 1 and obs.shape[2] > 16 and obs.shape[3] > 16:
        return obs[0, -1, :, :]

    # Single-channel fallback
    if obs.shape[-1] == 1:
        return obs[0, :, :, 0]
    if obs.shape[1] == 1:
        return obs[0, 0, :, :]

    raise ValueError(f"Could not infer grayscale frame layout from obs shape {obs.shape}")


def main() -> None:
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    output_dir = Path(args.output_dir)
    overlays_dir = output_dir / "overlays"
    gifs_dir = output_dir / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gifs:
        gifs_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(
        env_id=args.env_id,
        seed=args.seed,
        frame_stack=args.frame_stack,
        clip_reward=False,
        minimal_actions=args.minimal_actions,
        render_mode=None,
    )
    model = DQN.load(args.model_path, env=env)
    model.policy.set_training_mode(False)

    target_conv = find_last_conv_layer(model)
    cam_extractor = GradCamExtractor(target_conv)

    aggregated_columns: np.ndarray | None = None
    sample_count = 0
    episode_rewards: list[float] = []
    exported_gifs: list[str] = []

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = np.array([False])
            ep_reward = 0.0
            step_idx = 0
            ep_gif_frames: list[np.ndarray] = []

            while not bool(done[0]) and step_idx < args.max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs_next, rewards, done, _infos = env.step(action)
                ep_reward += float(rewards[0])

                should_sample = (step_idx % args.sample_every) == 0
                if should_sample:
                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    obs_tensor = obs_tensor.to(model.device)

                    model.policy.q_net.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        q_values = model.policy.q_net(obs_tensor)
                        action_idx = int(torch.argmax(q_values, dim=1).item())
                        q_values[0, action_idx].backward()

                    cam = cam_extractor.compute_cam()[0]

                    gray_frame = extract_latest_grayscale(obs)
                    cam_resized = cv2.resize(
                        cam,
                        dsize=(gray_frame.shape[1], gray_frame.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    cam_resized = np.clip(cam_resized, 0.0, 1.0)

                    col_hist = column_importance_from_cam(cam_resized)
                    if aggregated_columns is None:
                        aggregated_columns = np.zeros_like(col_hist, dtype=np.float64)
                    aggregated_columns += col_hist.astype(np.float64)
                    sample_count += 1

                    if args.save_overlays:
                        overlay = overlay_heatmap(gray_frame, cam_resized)
                        out_name = f"ep{ep:03d}_step{step_idx:05d}.png"
                        cv2.imwrite(
                            str(overlays_dir / out_name),
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                        )
                    else:
                        overlay = overlay_heatmap(gray_frame, cam_resized)

                    if args.save_gifs:
                        ep_gif_frames.append(overlay)

                obs = obs_next
                step_idx += 1

            episode_rewards.append(ep_reward)

            if args.save_gifs and ep_gif_frames:
                gif_path = gifs_dir / f"gradcam_episode_{ep:03d}.gif"
                imageio.mimsave(gif_path, ep_gif_frames, fps=args.gif_fps)
                exported_gifs.append(str(gif_path))

            print(
                f"Episode {ep + 1}/{args.episodes} reward={ep_reward:.2f} "
                f"steps={step_idx}"
            )

        if aggregated_columns is None or sample_count == 0:
            raise RuntimeError("No Grad-CAM samples collected. Try --sample-every 1.")

        aggregated_columns /= float(sample_count)
        aggregated_columns = aggregated_columns / (aggregated_columns.sum() + 1e-12)

        npy_path = output_dir / "x_axis_importance.npy"
        np.save(npy_path, aggregated_columns.astype(np.float32))

        csv_path = output_dir / "x_axis_importance.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_column", "importance"])
            for x_idx, value in enumerate(aggregated_columns.tolist()):
                writer.writerow([x_idx, value])

        hist_img_path = output_dir / "x_axis_importance_histogram.png"
        save_histogram_image(aggregated_columns.astype(np.float32), hist_img_path)

        top_k = 10
        top_indices = np.argsort(aggregated_columns)[::-1][:top_k]
        summary = {
            "model_path": str(args.model_path),
            "episodes": args.episodes,
            "sample_every": args.sample_every,
            "samples_collected": sample_count,
            "mean_reward": float(np.mean(np.asarray(episode_rewards, dtype=np.float32))),
            "top_columns": [
                {"x_column": int(i), "importance": float(aggregated_columns[i])}
                for i in top_indices
            ],
            "outputs": {
                "x_axis_importance_npy": str(npy_path),
                "x_axis_importance_csv": str(csv_path),
                "x_axis_importance_histogram_png": str(hist_img_path),
                "overlays_dir": str(overlays_dir) if args.save_overlays else None,
                "gradcam_gifs": exported_gifs if args.save_gifs else None,
            },
        }
        summary_path = output_dir / "gradcam_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\nGrad-CAM extraction complete")
        print(f"Samples collected: {sample_count}")
        print(f"Saved x-axis histogram: {csv_path}")
        print(f"Saved summary: {summary_path}")
    finally:
        cam_extractor.close()
        env.close()


if __name__ == "__main__":
    main()
