import gymnasium as gym
import numpy as np
from src.lib import wrappers
import matplotlib.pyplot as plt
import torch
from src.lib.dqn_model import DQN


ENV_NAME = "RiverraidNoFrameskip-v4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = wrappers.make_env(ENV_NAME, frameskip=4) #, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
# env = gym.make(ENV_NAME)
observation, info = env.reset()
print(f"Observation: {observation}")
print(f"Observation shape: {observation.shape}")
print(f"Observation dtype: {observation.dtype}")

print(f"Action space: {env.action_space}")

model_path = "/home/finnf/dev/deep_q_networks/models/RiverraidNoFrameskip-v4_ClippedReward_PrioReplay_WithFrameSkip_NoRepeatAction_NegTrmlRwd=-50_NoopMax=30_ReplaySize=250000_LearningRate=0.00025_EpsilonFinal=0.01_EpsilonDecayLastFrame=250000_RewardStartSize=50000_Gamma=0.99_BatchSize=32-best_-45.dat"
net = DQN(env.observation_space.shape, env.action_space.n)
state = torch.load(model_path, map_location=lambda stg, _: stg, weights_only=True)
net.load_state_dict(state)

actions = []
observations = []
rewards = []
# Take actions
for i in range(100):
    state_v = torch.tensor(np.expand_dims(observation, 0))
    q_vals = net(state_v).data.numpy()[0]
    action = int(np.argmax(q_vals))
    actions.append(action)
    observations.append(observation)
    observation, reward, is_done, is_trunc, _ = env.step(action)
    rewards.append(reward)
    if is_done or is_trunc:
        break

print(f"Actions: {actions}")

plt.figure(figsize=(15, 10))
num_obs = len(observations)
rows = (num_obs + 3) // 4 

# Create a grid of subplots
fig, axs = plt.subplots(rows, 4, figsize=(15, 3*rows))
axs = axs.flatten() if rows > 1 else axs 

for i in range(num_obs):
    ax = axs[i]
    ax.imshow(observations[i][0], cmap='gray')  # First frame of each observation
    ax.set_title(f'Obs {i}, Action: {actions[i]}, Reward: {rewards[i]}')
    ax.axis('off')

# Hide unused subplots
for i in range(num_obs, len(axs)):
    if rows > 1 or i < len(axs):
        axs[i].axis('off')

plt.tight_layout()
plt.show()

