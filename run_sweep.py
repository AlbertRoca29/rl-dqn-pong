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


def build_full_night_screen_runs(
    lr_values: list[float],
    buffer_values: list[int],
    final_eps_values: list[float],
    policy_values: list[str],
    run_prefix: str,
    ofat_warmup_condition: str,
    interaction_runs: int,
    greedy_lr: float | None,
    greedy_buffer: int | None,
    greedy_final_eps: float | None,
    greedy_policy: str | None,
) -> tuple[list[RunConfig], bool]:
    if not (
        len(lr_values) == len(buffer_values) == len(final_eps_values) == len(policy_values) == 3
    ):
        raise ValueError("full_night_screen expects exactly 3 values for each sweep dimension.")

    baseline = {
        "learning_rate": lr_values[1],
        "buffer_size": buffer_values[1],
        "exploration_final_eps": final_eps_values[1],
        "policy_variant": policy_values[1],
    }

    ofat_use_warmup = ofat_warmup_condition == "warm"
    runs: list[RunConfig] = []
    seen_run_names: set[str] = set()

    def cfg_tag(cfg: dict) -> str:
        lr = f"lr{cfg['learning_rate']:.0e}".replace("+", "")
        buf = f"buf{cfg['buffer_size'] // 1000}k"
        eps = f"feps{str(cfg['exploration_final_eps']).replace('.', 'p')}"
        pol = f"policy{cfg['policy_variant']}"
        return f"{lr}_{buf}_{eps}_{pol}"

    def add_run(cfg: dict, use_warmup: bool, minimal_actions: bool, experiment_label: str) -> None:
        warm_label = "warmup_on" if use_warmup else "warmup_off"
        min_label = "minimal_actions_on" if minimal_actions else "minimal_actions_off"
        run_name = (
            f"{run_prefix}_{experiment_label}_{warm_label}_{min_label}_{cfg_tag(cfg)}"
        )
        if run_name in seen_run_names:
            return
        seen_run_names.add(run_name)
        runs.append(
            RunConfig(
                run_name=run_name,
                learning_rate=cfg["learning_rate"],
                buffer_size=cfg["buffer_size"],
                exploration_final_eps=cfg["exploration_final_eps"],
                policy_variant=cfg["policy_variant"],
                use_warmup=use_warmup,
                minimal_actions=minimal_actions,
            )
        )

    # 1) Warm-up impact at baseline: one pair (warm vs cold), minimal-actions ON.
    add_run(baseline, True, True, "q1_warmup_effect_baseline_pair")
    add_run(baseline, False, True, "q1_warmup_effect_baseline_pair")

    # 2) OFAT sensitivity: vary one factor at a time using a single warm-up condition.
    factors = {
        "learning_rate": lr_values,
        "buffer_size": buffer_values,
        "exploration_final_eps": final_eps_values,
        "policy_variant": policy_values,
    }
    for factor_name, values in factors.items():
        for idx, level_tag in ((0, "low"), (2, "high")):
            cfg = dict(baseline)
            cfg[factor_name] = values[idx]
            add_run(
                cfg,
                ofat_use_warmup,
                True,
                f"q2_ofat_sensitivity_{factor_name}_{level_tag}",
            )

    # 3) Minimal-actions impact: one pair at baseline under the OFAT warm-up condition.
    add_run(
        baseline,
        ofat_use_warmup,
        False,
        "q3_minimal_actions_effect_baseline_pair",
    )

    # 4) Greedy best-combo run (optional): user provides best levels from OFAT.
    greedy_added = False
    if (
        greedy_lr is not None
        and greedy_buffer is not None
        and greedy_final_eps is not None
        and greedy_policy is not None
    ):
        if greedy_policy not in policy_values:
            raise ValueError(
                f"greedy-policy must be one of {policy_values}, got: {greedy_policy}"
            )
        greedy_cfg = {
            "learning_rate": greedy_lr,
            "buffer_size": greedy_buffer,
            "exploration_final_eps": greedy_final_eps,
            "policy_variant": greedy_policy,
        }
        add_run(greedy_cfg, ofat_use_warmup, True, "q4_greedy_best_combo")
        greedy_added = True

    # 5) Targeted interaction runs: policy size x learning-rate.
    if interaction_runs >= 1:
        cfg_high_lr_large = dict(baseline)
        cfg_high_lr_large["learning_rate"] = lr_values[2]
        cfg_high_lr_large["policy_variant"] = policy_values[2]
        add_run(
            cfg_high_lr_large,
            ofat_use_warmup,
            True,
            "q5_interaction_policy_large_lr_high",
        )

    if interaction_runs >= 2:
        cfg_low_lr_large = dict(baseline)
        cfg_low_lr_large["learning_rate"] = lr_values[0]
        cfg_low_lr_large["policy_variant"] = policy_values[2]
        add_run(
            cfg_low_lr_large,
            ofat_use_warmup,
            True,
            "q5_interaction_policy_large_lr_low",
        )

    return runs, greedy_added


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

    parser.add_argument(
        "--design",
        choices=["screen", "full", "full_night_screen"],
        default="screen",
    )
    parser.add_argument(
        "--full-night-ofat-warmup",
        choices=["warm", "cold"],
        default="warm",
        help="For full_night_screen: warm-up condition used in OFAT, minimal-actions pair, and interaction runs.",
    )
    parser.add_argument(
        "--interaction-runs",
        type=int,
        choices=[1, 2],
        default=2,
        help="For full_night_screen: number of targeted policy-size x LR interaction runs.",
    )
    parser.add_argument(
        "--greedy-lr",
        type=float,
        default=None,
        help="For full_night_screen: greedy best-combo learning rate from OFAT findings.",
    )
    parser.add_argument(
        "--greedy-buffer",
        type=int,
        default=None,
        help="For full_night_screen: greedy best-combo replay buffer size from OFAT findings.",
    )
    parser.add_argument(
        "--greedy-final-eps",
        type=float,
        default=None,
        help="For full_night_screen: greedy best-combo final epsilon from OFAT findings.",
    )
    parser.add_argument(
        "--greedy-policy",
        type=str,
        default=None,
        help="For full_night_screen: greedy best-combo policy variant from OFAT findings.",
    )
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

    greedy_added = False
    if args.design == "full_night_screen":
        run_list, greedy_added = build_full_night_screen_runs(
            lr_values,
            buffer_values,
            final_eps_values,
            policy_values,
            args.run_prefix,
            args.full_night_ofat_warmup,
            args.interaction_runs,
            args.greedy_lr,
            args.greedy_buffer,
            args.greedy_final_eps,
            args.greedy_policy,
        )
    else:
        if args.design == "screen":
            design = build_screen_design(lr_values, buffer_values, final_eps_values, policy_values)
        else:
            design = build_full_design(lr_values, buffer_values, final_eps_values, policy_values)

        run_list = list(
            iter_runs(
                design,
                args.run_prefix,
                args.warmup_conditions,
                args.minimal_actions_mode,
            )
        )

    needs_warmup_buffer = args.design == "full_night_screen" or args.warmup_conditions in {
        "warm",
        "both",
    }
    if needs_warmup_buffer:
        warmup_path = Path(args.warmup_buffer)
        if not warmup_path.exists():
            raise FileNotFoundError(
                f"Warm-up buffer not found: {warmup_path}. "
                "Create it first with --warmup-only in train_dqn_pong.py"
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
        "full_night_ofat_warmup": args.full_night_ofat_warmup,
        "interaction_runs": args.interaction_runs,
        "greedy_run_included": greedy_added,
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
    if args.design == "full_night_screen" and not greedy_added:
        print(
            "Note: greedy best-combo run not added. Provide --greedy-lr, --greedy-buffer, "
            "--greedy-final-eps, and --greedy-policy after OFAT analysis."
        )

    if args.dry_run:
        for i, run_cfg in enumerate(run_list, start=1):
            print(f"[{i:03d}] {run_cfg.run_name}")
        return

    completed = 0
    failed = 0
    skipped = 0

    for i, run_cfg in enumerate(run_list, start=1):
        # Check if run has already been completed
        run_dir = Path(args.log_dir) / run_cfg.run_name
        if run_dir.exists():
            skipped += 1
            print(f"\n[{i:03d}/{len(run_list):03d}] Skipping {run_cfg.run_name} (already completed)")
            continue

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
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
