"""Environment factories and wrappers for training/testing DQN on Pong."""

from __future__ import annotations

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium as gym


DEFAULT_ENV_ID = "PongNoFrameskip-v4"


def make_train_env(
    env_id: str = DEFAULT_ENV_ID,
    n_envs: int = 1,
    seed: int = 0,
    frame_stack: int = 4,
    clip_reward: bool = True,
):
    """Create a vectorized Atari env for training.

    Uses AtariWrapper for preprocessing (grayscale, resize to 84x84,
    frame skipping/max-pooling, optional reward clipping) and stacks
    consecutive observations to provide temporal context.
    """
    vec_env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs={"clip_reward": clip_reward},
    )
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


def make_eval_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    frame_stack: int = 4,
    clip_reward: bool = False,
    render_mode: str | None = None,
):
    """Create a single-environment vec env for deterministic evaluation.

    Reward clipping is disabled by default so reported returns match the task
    metric in the original game reward scale.
    """

    def _make_single_env():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(env, clip_reward=clip_reward)
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_make_single_env])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env
