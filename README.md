# Activity 1 - Reinforcement Learning (DQN on Pong)

This project implements both parts of the activity:
1. Train a DQN agent on Pong.
2. Evaluate the trained model on 100 episodes and export best/worst episode animations.

## Project files

- `wrappers.py`: environment factory functions and preprocessing wrappers.
- `train_dqn_pong.py`: DQN training script.
- `evaluate_dqn_pong.py`: 100-episode evaluation + GIF export.
- `activity_report.md`: written answers/discussion for the assignment.
- `requirements.txt`: required dependencies to run the project.

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## 2) Train a baseline model

```bash
python train_dqn_pong.py \
  --run-name dqn_baseline \
  --total-timesteps 1000000 \
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

## 4) Evaluate a trained model on 100 episodes

```bash
python evaluate_dqn_pong.py \
  --model-path runs/dqn_baseline/models/best_model.zip \
  --episodes 100 \
  --output-dir evaluation_baseline
```

Outputs:
- `evaluation_baseline/evaluation_summary.json`
- `evaluation_baseline/worst_episode.gif`
- `evaluation_baseline/best_episode.gif`

## Notes

- For a strong Pong score with DQN, training usually needs more than 1M timesteps (often several million).
- If your machine is slow, start with fewer timesteps for debugging and then launch long runs.
- If you have ROM/environment issues, reinstall Atari extras:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```
