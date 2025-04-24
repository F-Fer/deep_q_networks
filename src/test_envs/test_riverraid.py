import gymnasium as gym
from src.lib import wrappers
import matplotlib.pyplot as plt

ENV_NAME = "RiverraidNoFrameskip-v4"

# env = wrappers.make_env(ENV_NAME) #, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
env = gym.make(ENV_NAME)
observation, info = env.reset()
print(f"Observation: {observation}")
print(f"Observation shape: {observation.shape}")
print(f"Observation dtype: {observation.dtype}")

print(f"Action space: {env.action_space}")

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"Info: {info}")

# Take 100 random actions
for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    ax = axs[i//2, i%2]
    ax.imshow(observation[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Frame {i}')
plt.tight_layout()
plt.show()
