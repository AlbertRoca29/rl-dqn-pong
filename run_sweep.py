"""Run overnight DQN sweeps with optional warm-up replay buffer reuse.

Design modes:
- full: all combinations
- screen: reduced design (baseline + one-factor-at-a-time extremes)
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class RunConfig:
    run_name: str
    learning_rate: float
    buffer_size: int
    exploration_final_eps: float
    policy_variant: str
    use_warmup: bool
    minimal_actions: bool


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_full_design(
    lr_values: list[float],
    buffer_values: list[int],
    final_eps_values: list[float],
    policy_values: list[str],
) -> list[dict]:
    design = []
    for lr, buffer_size, final_eps, policy in itertools.product(
        lr_values, buffer_values, final_eps_values, policy_values
    ):
        design.append(
            {
                "learning_rate": lr,
                "buffer_size": buffer_size,
                "exploration_final_eps": final_eps,
                "policy_variant": policy,
            }
        )
    return design


def build_screen_design(
    lr_values: list[float],
    buffer_values: list[int],
    final_eps_values: list[float],
    policy_values: list[str],
) -> list[dict]:
    if not (
        len(lr_values) == len(buffer_values) == len(final_eps_values) == len(policy_values) == 3
    ):
        raise ValueError("screen design expects exactly 3 values for each sweep dimension.")

    baseline = {
        "learning_rate": lr_values[1],
        "buffer_size": buffer_values[1],
        "exploration_final_eps": final_eps_values[1],
        "policy_variant": policy_values[1],
    }

    design: list[dict] = [baseline]
    factors = {
        "learning_rate": lr_values,
        "buffer_size": buffer_values,
        "exploration_final_eps": final_eps_values,
        "policy_variant": policy_values,
    }

    for factor_name, values in factors.items():
        for idx in (0, 2):
            cfg = dict(baseline)
            cfg[factor_name] = values[idx]
            design.append(cfg)

    # Remove accidental duplicates while preserving order.
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for cfg in design:
        key = (
            cfg["learning_rate"],
            cfg["buffer_size"],
            cfg["exploration_final_eps"],
            cfg["policy_variant"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    return deduped


def make_run_name(prefix: str, cfg: dict, use_warmup: bool) -> str:
    lr_tag = f"lr{cfg['learning_rate']:.0e}".replace("+", "")
    buf_tag = f"buf{cfg['buffer_size'] // 1000}k"
    eps_tag = f"feps{str(cfg['exploration_final_eps']).replace('.', 'p')}"
    pol_tag = f"pol{cfg['policy_variant']}"
    warm_tag = "warm" if use_warmup else "cold"
    return f"{prefix}_{lr_tag}_{buf_tag}_{eps_tag}_{pol_tag}_{warm_tag}"


def iter_runs(
    design: list[dict],
    run_prefix: str,
    warmup_conditions: str,
    minimal_actions_mode: str,
) -> Iterable[RunConfig]:
    if warmup_conditions == "warm":
        warm_values = [True]
    elif warmup_conditions == "cold":
        warm_values = [False]
    else:
        warm_values = [True, False]

    if minimal_actions_mode == "on":
        min_values = [True]
    elif minimal_actions_mode == "off":
        min_values = [False]
    elif minimal_actions_mode == "both":
        min_values = [True, False]
    else:
        # ablation: run all with minimal-actions ON, and only baseline with OFF.
        min_values = [True]

    baseline = design[0] if design else None

    for cfg in design:
        for use_warmup in warm_values:
            for minimal_actions in min_values:
                run_name = make_run_name(run_prefix, cfg, use_warmup)
                run_name = f"{run_name}_{'minon' if minimal_actions else 'minoff'}"
                yield RunConfig(
                    run_name=run_name,
                    learning_rate=cfg["learning_rate"],
                    buffer_size=cfg["buffer_size"],
                    exploration_final_eps=cfg["exploration_final_eps"],
                    policy_variant=cfg["policy_variant"],
                    use_warmup=use_warmup,
                    minimal_actions=minimal_actions,
                )

            if minimal_actions_mode == "ablation" and baseline is not None and cfg == baseline:
                run_name = make_run_name(run_prefix, cfg, use_warmup)
                run_name = f"{run_name}_minoff"
                yield RunConfig(
                    run_name=run_name,
                    learning_rate=cfg["learning_rate"],
                    buffer_size=cfg["buffer_size"],
                    exploration_final_eps=cfg["exploration_final_eps"],
                    policy_variant=cfg["policy_variant"],
                    use_warmup=use_warmup,
                    minimal_actions=False,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN sweep experiments")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=str, default="train_dqn_pong.py")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-prefix", type=str, default=f"night_{datetime.now():%Y%m%d}")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--minimal-actions-mode",
        choices=["on", "off", "both", "ablation"],
        default="ablation",
        help=(
            "How to include minimal-actions factor: on/off for all runs, both for full pairwise, "
            "or ablation (all runs on + baseline off)."
        ),
    )

    parser.add_argument("--lr-values", type=str, default="5e-5,1e-4,2e-4")
    parser.add_argument("--buffer-values", type=str, default="100000,200000,400000")
    parser.add_argument("--final-eps-values", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--policy-values", type=str, default="small,base,large")

    parser.add_argument("--design", choices=["screen", "full"], default="screen")
    parser.add_argument(
        "--warmup-conditions",
        choices=["both", "warm", "cold"],
        default="both",
        help="Run sweeps with warm-up replay buffer, without it, or both.",
    )
    parser.add_argument("--warmup-buffer", type=str, default="runs/shared/warmup_50k.pkl")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lr_values = parse_csv_floats(args.lr_values)
    buffer_values = parse_csv_ints(args.buffer_values)
    final_eps_values = parse_csv_floats(args.final_eps_values)
    policy_values = parse_csv_strings(args.policy_values)

    if args.design == "screen":
        design = build_screen_design(lr_values, buffer_values, final_eps_values, policy_values)
    else:
        design = build_full_design(lr_values, buffer_values, final_eps_values, policy_values)

    if args.warmup_conditions in {"warm", "both"}:
        warmup_path = Path(args.warmup_buffer)
        if not warmup_path.exists():
            raise FileNotFoundError(
                f"Warm-up buffer not found: {warmup_path}. "
                "Create it first with --warmup-only in train_dqn_pong.py"
            )

    run_list = list(
        iter_runs(
            design,
            args.run_prefix,
            args.warmup_conditions,
            args.minimal_actions_mode,
        )
    )
    if args.max_runs > 0:
        run_list = run_list[: args.max_runs]

    sweep_dir = Path(args.log_dir) / f"{args.run_prefix}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "design": args.design,
        "warmup_conditions": args.warmup_conditions,
        "total_runs": len(run_list),
        "defaults": {
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "learning_starts": args.learning_starts,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "minimal_actions_mode": args.minimal_actions_mode,
        },
        "runs": [asdict(r) for r in run_list],
    }
    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Planned runs: {len(run_list)}")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        for i, run_cfg in enumerate(run_list, start=1):
            print(f"[{i:03d}] {run_cfg.run_name}")
        return

    completed = 0
    failed = 0

    for i, run_cfg in enumerate(run_list, start=1):
        cmd = [
            args.python,
            args.train_script,
            "--run-name",
            run_cfg.run_name,
            "--log-dir",
            args.log_dir,
            "--total-timesteps",
            str(args.total_timesteps),
            "--seed",
            str(args.seed),
            "--learning-rate",
            str(run_cfg.learning_rate),
            "--buffer-size",
            str(run_cfg.buffer_size),
            "--learning-starts",
            str(args.learning_starts),
            "--exploration-final-eps",
            str(run_cfg.exploration_final_eps),
            "--policy-variant",
            run_cfg.policy_variant,
            "--eval-freq",
            str(args.eval_freq),
            "--eval-episodes",
            str(args.eval_episodes),
        ]

        if run_cfg.minimal_actions:
            cmd.append("--minimal-actions")
        else:
            cmd.append("--no-minimal-actions")

        if run_cfg.use_warmup:
            cmd.extend(["--load-replay-buffer", args.warmup_buffer])

        print(f"\n[{i:03d}/{len(run_list):03d}] Running {run_cfg.run_name}")
        print(" ".join(cmd))

        log_path = logs_dir / f"{run_cfg.run_name}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

        if proc.returncode == 0:
            completed += 1
            print(f"OK -> {log_path}")
        else:
            failed += 1
            print(f"FAILED (code={proc.returncode}) -> {log_path}")
            if not args.continue_on_error:
                break

    print("\nSweep finished")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
