import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeReward
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)
    

class NegativeTerminalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, terminal_reward: float = -1.0):
        super().__init__(env)
        self.terminal_reward = terminal_reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            reward += self.terminal_reward
        return obs, reward, done, truncated, info
    

class ScaleRewardWrapper(gym.RewardWrapper):
    """Min-max scaling of rewards to [-1, 1]"""
    def __init__(self, env, min_val: float, max_val: float):
        super().__init__(env)
        self.min_val = min_val
        self.max_val = max_val

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        scaled_reward = 2 * ((reward - self.min_val) / (self.max_val - self.min_val)) - 1
        return obs, scaled_reward, done, truncated, info
    

class KeepAliveReward(gym.RewardWrapper):
    """Add small reward for each step, which not leads to game over"""
    def __init__(self, env, reward: float = 1):
        super().__init__(env)
        self.reward = reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        new_reward = reward
        if not done and not truncated:  
            new_reward += self.reward
        return obs, new_reward, done, truncated, info
    

def make_env(env_name: str, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=True, noop_max=30, screen_size=84)
    env = NegativeTerminalRewardWrapper(env, terminal_reward=-10.0)
    # env = KeepAliveReward(env)
    # env = ScaleRewardWrapper(env, -100, 100)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env
