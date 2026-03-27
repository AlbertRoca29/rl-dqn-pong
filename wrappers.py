"""Environment factories and wrappers for training/testing DQN on Pong."""

from __future__ import annotations

from functools import partial

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
import gymnasium as gym


DEFAULT_ENV_ID = "PongNoFrameskip-v4"


class PongMinimalActionWrapper(gym.ActionWrapper):
    """Reduce Pong action space to no-op and two movement directions.

    This often improves sample efficiency because the agent does not need to
    explore redundant action variants (e.g., FIRE combinations in Pong).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()

        if "RIGHT" in meanings and "LEFT" in meanings:
            move_actions = ("RIGHT", "LEFT")
        elif "UP" in meanings and "DOWN" in meanings:
            move_actions = ("UP", "DOWN")
        else:
            raise ValueError(
                f"Unsupported Pong action meanings: {meanings}. "
                "Could not find movement actions."
            )

        self._action_map = [
            meanings.index("NOOP"),
            meanings.index(move_actions[0]),
            meanings.index(move_actions[1]),
        ]
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def action(self, action):
        return self._action_map[int(action)]


def _atari_preprocess_wrapper(
    env: gym.Env,
    *,
    clip_reward: bool,
    minimal_actions: bool,
) -> gym.Env:
    if minimal_actions and env.spec is not None and "Pong" in env.spec.id:
        env = PongMinimalActionWrapper(env)
    return AtariWrapper(env, clip_reward=clip_reward)


def make_train_env(
    env_id: str = DEFAULT_ENV_ID,
    n_envs: int = 1,
    seed: int = 0,
    frame_stack: int = 4,
    clip_reward: bool = True,
    minimal_actions: bool = False,
    vec_env_type: str = "dummy",
):
    """Create a vectorized Atari env for training.

    Uses AtariWrapper for preprocessing (grayscale, resize to 84x84,
    frame skipping/max-pooling, optional reward clipping) and stacks
    consecutive observations to provide temporal context.
    """
    vec_env_cls = SubprocVecEnv if vec_env_type == "subproc" else DummyVecEnv
    vec_env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        wrapper_class=partial(
            _atari_preprocess_wrapper,
            clip_reward=clip_reward,
            minimal_actions=minimal_actions,
        ),
    )
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env


def make_eval_env(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = 0,
    frame_stack: int = 4,
    clip_reward: bool = False,
    minimal_actions: bool = False,
    render_mode: str | None = None,
):
    """Create a single-environment vec env for deterministic evaluation.

    Reward clipping is disabled by default so reported returns match the task
    metric in the original game reward scale.
    """

    def _make_single_env():
        env = gym.make(env_id, render_mode=render_mode)
        env = _atari_preprocess_wrapper(
            env,
            clip_reward=clip_reward,
            minimal_actions=minimal_actions,
        )
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_make_single_env])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    return vec_env
