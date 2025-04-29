import gymnasium as gym
from src.lib import wrappers
import matplotlib.pyplot as plt
import torch

ENV_NAME = "RiverraidNoFrameskip-v4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = wrappers.make_env(ENV_NAME, frameskip=4) #, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
# env = gym.make(ENV_NAME)
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

model_path = "models/RiverraidNoFrameskip-v4_ClippedReward_NoKeepAlive_WithFrameSkip_NoRepeatAction_NegativeTerminalReward=-10_ReplaySize=1000000_LearningRate=0.00025_EpsilonFinal=0.1_EpsilonDecayLastFrame=1000000_RewardStartSize=50000_Gamma=0.99_BatchSize=32-best_-7.dat"
model = torch.load(model_path)

# Take actions
for i in range(100):
    state_v = torch.as_tensor(observation).to(device)
    state_v.unsqueeze_(0)
    q_vals_v = model(state_v)
    _, act_v = torch.max(q_vals_v, dim=1)
    action = int(act_v.item())
    observation, reward, terminated, truncated, info = env.step(action)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    ax = axs[i//2, i%2]
    ax.imshow(observation[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Frame {i}')
plt.tight_layout()
plt.show()
