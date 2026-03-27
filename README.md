# Activity 1 - Reinforcement Learning (DQN on Pong)

This project implements both parts of the activity:
1. Train a DQN agent on Pong.
2. Evaluate the trained model on 100 episodes and export best/worst episode animations.

## Project files

- `wrappers.py`: environment factory functions and preprocessing wrappers.
- `train_dqn_pong.py`: DQN training script.
- `evaluate_dqn_pong.py`: evaluation + extracting visual insights.
- `requirements.txt`: required dependencies to run the project.

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 2) Train a baseline model

```bash
python train_dqn_pong.py \
  --run-name dqn_baseline \
  --total-timesteps 1000000 \
  --minimal-actions \
  --eval-freq 10000 \
  --eval-episodes 10
```

Outputs are saved under `runs/dqn_baseline/`:
- `models/final_model.zip`
- `models/best_model.zip` (best evaluation checkpoint)
- TensorBoard logs in `tb/`

To visualize training:

```bash
tensorboard --logdir runs
```

## 3) Hyperparameter experiments

Run several experiments with different configurations (example):

```bash
python train_dqn_pong.py --run-name dqn_lr1e4 --learning-rate 1e-4 --total-timesteps 1000000
python train_dqn_pong.py --run-name dqn_lr5e5 --learning-rate 5e-5 --total-timesteps 1000000
python train_dqn_pong.py --run-name dqn_eps005 --exploration-fraction 0.05 --total-timesteps 1000000
python train_dqn_pong.py --run-name dqn_buf200k --buffer-size 200000 --total-timesteps 1000000
```

Use `activity_report.md` to document and compare these runs.

### Optional: Reuse a 50k warm-up buffer for faster sweeps

If you run many experiments with the same environment setup and
`--learning-starts 50000`, you can pre-generate that warm-up once:

```bash
python train_dqn_pong.py \
  --run-name warmup_only \
  --learning-starts 50000 \
  --warmup-only \
  --save-replay-buffer runs/shared/warmup_50k.pkl
```

Then launch experiments that load it and start training immediately:

```bash
python train_dqn_pong.py \
  --run-name dqn_lr1e4_warm \
  --learning-rate 1e-4 \
  --learning-starts 50000 \
  --load-replay-buffer runs/shared/warmup_50k.pkl
```

Notes for report quality:
- This is great for fast screening and iteration.
- For final conclusions, rerun best settings from scratch (no shared warm-up)
  and with multiple seeds to avoid bias from one initial replay dataset.

### Overnight multi-experiment sweep

Use `run_sweep.py` to launch many runs while you are away.

It supports two designs:
- `screen`: reduced set (baseline + one-factor-at-a-time extremes).
- `full`: full factorial (`3 x 3 x 3 x 3 x 2 = 162` runs if using warm + cold).

And four minimal-actions modes:
- `on`: all runs with `--minimal-actions`.
- `off`: all runs without `--minimal-actions`.
- `both`: duplicate every run with and without minimal-actions.
- `ablation`: all runs with minimal-actions + only baseline without it (cheap effect estimate).

First, create the shared warm-up buffer (once):

```bash
python train_dqn_pong.py \
  --run-name warmup_only \
  --learning-starts 50000 \
  --warmup-only \
  --minimal-actions \
  --save-replay-buffer runs/shared/warmup_50k.pkl
```

Then run a reduced overnight sweep (recommended):

```bash
python run_sweep.py \
  --design screen \
  --warmup-conditions both \
  --minimal-actions-mode ablation \
  --total-timesteps 300000 \
  --run-prefix night_screen
```

To inspect planned runs without starting them:

```bash
python run_sweep.py --design screen --warmup-conditions both --minimal-actions-mode ablation --dry-run
```


## 4) Evaluate a trained model on 100 episodes

```bash
python evaluate_dqn_pong.py \
  --model-path runs/dqn_baseline/models/best_model.zip \
  --episodes 100 \
  --minimal-actions \
  --output-dir evaluation_baseline
```

Outputs:
- `evaluation_baseline/evaluation_summary.json`
- `evaluation_baseline/worst_episode.gif`
- `evaluation_baseline/best_episode.gif`


## Notes

- `--minimal-actions` is an environment-side optimization for Pong: it reduces
  the action space to no-op + two movement actions, which often improves
  sample efficiency without changing DQN hyperparameters.
- For a strong Pong score with DQN, training usually needs more than 1M timesteps (often several million).
- If your machine is slow, start with fewer timesteps for debugging and then launch long runs.
- If you have ROM/environment issues, reinstall Atari extras:

<!-- ```bash
pip install "gymnasium[atari,accept-rom-license]"
``` -->
