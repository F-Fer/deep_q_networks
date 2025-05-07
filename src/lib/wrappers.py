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


class FuelRewardWrapper(gym.RewardWrapper):
    """
    Adds a reward when the agent collects fuel in Riverraid.
    Uses ALE to access the game's RAM and monitor fuel level changes.
    """
    def __init__(self, env, reward: float = 1.0):
        super().__init__(env)
        self.fuel_reward = reward
        # The fuel level in Riverraid is stored at RAM address 120 (0x78)
        self.fuel_address = 120
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if not done and not truncated:
            ram = self.env.unwrapped.ale.getRAM()
            entered_fuel_deposit = ram[self.fuel_address] == 3
            if entered_fuel_deposit:
                reward += self.fuel_reward
        return obs, reward, done, truncated, info


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper to go from 18 actions to 5 actions.
    """

    KEY_ACTION_MAP = [
        2,   # Fire
        8,   # Move right
        4,   # Move left
        5,   # Do nothing/slow down
        1,   # Alternative fire
    ]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.KEY_ACTION_MAP))

    def step(self, action):
        return self.env.step(self.KEY_ACTION_MAP[action])

class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, visitation_counts=None, bonus_scale=0.01):
        super().__init__(env)
        # Use a hash table to store state visitation counts
        self.visitation_counts = {} if visitation_counts is None else visitation_counts
        self.bonus_scale = bonus_scale
        
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        
        state_hash = self._hash_observation(obs)
        
        # Update visitation count
        if state_hash in self.visitation_counts:
            self.visitation_counts[state_hash] += 1
        else:
            self.visitation_counts[state_hash] = 1
        
        # Calculate novelty bonus (higher for less-visited states)
        count = self.visitation_counts[state_hash]
        novelty_bonus = self.bonus_scale / np.sqrt(count)
        
        # Add to reward
        augmented_reward = reward + novelty_bonus
        
        return obs, augmented_reward, done, trunc, info
    
    def _hash_observation(self, obs):
        # Convert high-dimensional observation to a simpler hash
        # e.g., subsample or create a feature vector
        # For Atari, a simple approach is to downsample and binarize
        downsampled = obs[0, ::4, ::4]  # Take first channel and downsample
        binary = (downsampled > 128).astype(int)  # Binarize
        return hash(binary.tobytes())  # Create hash
    

class StickyActionsWrapper(gym.Wrapper):
    """
    Implements sticky actions from the paper "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents" by Machado et al.
    
    At each step, with probability p, the environment will ignore the agent's action
    and repeat the previous action instead.
    """
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0
        
    def reset(self, **kwargs):
        self.last_action = 0  # Reset to NOOP
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # With probability p, repeat the last action instead
        if np.random.random() < self.p:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)
    

class RandomStartWrapper(gym.Wrapper):
    """
    Start episodes with one shoot action followed by a random number of no-ops.
    This adds stochasticity to the initial state of the environment.
    """
    def __init__(self, env, max_random_frames=30):
        super().__init__(env)
        self.max_random_frames = max_random_frames
        self.noop_action = 0  # NOOP action
        self.shoot_action = 2  # SHOOT action
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Apply random number of no-op actions at the start
        if self.max_random_frames > 0:
            # First step is a shoot action
            obs, _, done, truncated, info = self.env.step(self.shoot_action)
            # Then random number of no-op actions
            num_random_frames = np.random.randint(0, self.max_random_frames)
            for _ in range(num_random_frames):
                obs, _, done, truncated, info = self.env.step(self.noop_action)
                if done or truncated:
                    obs, info = self.env.reset(**kwargs)
                    
        return obs, info



def make_env(
        env_name: str, 
        frameskip: int = 4, 
        sticky_action_prob: float = 0.0, 
        random_start_frames: int = 0, 
        screen_size: int = 84, 
        terminal_reward: float = -100.0,
        fuel_reward: float = 0.1,
        **kwargs
    ):
    """
    Create environment with wrappers for non-determinism and preprocessing.
    
    Args:
        env_name: Name of the Atari environment
        frameskip: Number of frames to skip between actions
        sticky_action_prob: Probability of repeating the previous action (stickiness)
        random_start_frames: Maximum number of no-op actions at the start of an episode
        screen_size: Size of the screen to downsample to
    """
    env = gym.make(env_name, **kwargs)
    
    # Standard Atari preprocessing
    env = atari_wrappers.AtariWrapper(
        env, 
        clip_reward=True, 
        noop_max=0,
        screen_size=screen_size,
        frame_skip=frameskip,
        action_repeat_probability=sticky_action_prob
    )
    
    # Add non-determinism
    env = RandomStartWrapper(env, max_random_frames=random_start_frames)
    
    # Reward shaping
    env = NegativeTerminalRewardWrapper(env, terminal_reward=terminal_reward)
    env = FuelRewardWrapper(env, reward=fuel_reward)
    
    # Action space and observation formatting
    env = ActionMaskWrapper(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    
    return env
