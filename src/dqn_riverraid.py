import gymnasium as gym
from lib import dqn_model
from lib import wrappers
from lib.noisy_dqn_model import NoisyDQN
import typing as tt

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter

DEFAULT_ENV_NAME = "RiverraidNoFrameskip-v4"
DEFAULT_MODEL_DIR = "models"
MEAN_REWARD_BOUND = 10_000
MAX_FRAMES = 3_000_000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 250_000
LEARNING_RATE = 0.00025 #1e-4
SYNC_TARGET_FRAMES = 10_000
REPLAY_START_SIZE = 50_000
FRAMESKIP = 2

# Parameters for the epsilon-greedy exploration
EPSILON_DECAY_LAST_FRAME = 250_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
# How often to reset noise in the noisy layers
NOISE_RESET_FRAMES = 1000

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,           # current state
    torch.LongTensor,           # actions
    torch.Tensor,               # rewards
    torch.BoolTensor,           # done || trunc
    torch.ByteTensor            # next state
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    """
    Experience buffer with prioritized experience replay.
    """
    def __init__(self, capacity: int, alpha=0.6):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.pos = 0
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        # Set priority to max priority for new experiences
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.priorities) < self.capacity:
            self.priorities = np.append(self.priorities, max_priority)
        else:
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta=0.4) -> tt.Tuple[tt.List[Experience], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        priorities = np.maximum(priorities, 1e-8)  # Ensure no zeros
        if np.any(np.isnan(priorities)):
            priorities = np.ones_like(priorities)  # Replace NaNs with 1s
        
        # Sample using priorities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Safety check to prevent NaN or infinite values
            if np.isnan(priority) or np.isinf(priority):
                priority = 1.0
            self.priorities[idx] = max(priority, 1e-8)  

class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, device: torch.device,
                  epsilon: float = 0.0, use_noisy: bool = False) -> tt.Optional[float]:
        done_reward = None

        if (not use_noisy and np.random.random() < epsilon) or (use_noisy and np.random.random() < epsilon):
            action = env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
           dones_t.to(device),  new_states_t.to(device)


def calc_loss(batch, indices, weights, net, tgt_net, device):
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)
    
    # Get current state-action values
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    
    with torch.no_grad():
        # Get next state values using Double DQN approach
        next_actions = net(new_states_t).argmax(dim=1)
        next_state_values = tgt_net(new_states_t).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[dones_t] = 0.0
    
    # Calculate expected state-action values
    expected_state_action_values = rewards_t + GAMMA * next_state_values
    
    # Convert weights to tensor and calculate weighted MSE loss
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    td_errors = expected_state_action_values - state_action_values
    
    # Clamp TD errors to prevent extreme values (optional, but helps stability)
    td_errors_clipped = torch.clamp(td_errors, -1.0, 1.0)
    
    # Calculate weighted loss
    loss = (weights_tensor * td_errors_clipped.pow(2)).mean()
    
    # For priority updates, use non-clipped TD errors
    with torch.no_grad():
        td_abs_errors = torch.abs(td_errors).detach().cpu().numpy()
    
    return loss, td_abs_errors


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--noisy", action="store_true", help="Use NoisyDQN instead of standard DQN")
    args = parser.parse_args()
    device = torch.device(args.dev)
    
    # Get use_noisy flag from args
    use_noisy = args.noisy
    
    # Adjust epsilon parameters if using noisy networks
    epsilon_start = 0.1 if use_noisy else EPSILON_START
    epsilon_final = 0.0 if use_noisy else EPSILON_FINAL

    # Define unique identifier for this parameter combination
    model_type = "NoisyDQN" if use_noisy else "DQN"
    env_changes = f"{model_type}_ClipRwd_PrioReplay_WithFrmSkip_NoRptAction_FuelReward_NegTrmlRwd=-100_NoopMax=30_ActionMask"
    param_id = f"FrmSkip={FRAMESKIP}_RplSize={REPLAY_SIZE}_LR={LEARNING_RATE}_EpsFinal={epsilon_final}_EpsDecayLastFrame={EPSILON_DECAY_LAST_FRAME}_RedStartSize={REPLAY_START_SIZE}_Gamma={GAMMA}_BatchSize={BATCH_SIZE}"
    param_id = env_changes + "_" + param_id
    print(f"Running {env_changes} with {param_id}")
    print(f"Length of param_id: {len(param_id)}")

    # Initialize environment
    env = wrappers.make_env(args.env, frameskip=FRAMESKIP)
    
    # Initialize network based on use_noisy
    if use_noisy:
        net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    writer = SummaryWriter(comment="-" + param_id)
    print(net)

    # Initialize experience buffer
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = epsilon_start

    # Initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Initialize total rewards
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    # Inner loop of parameter grid
    for frame_idx in range(MAX_FRAMES):
        # Update epsilon for epsilon-greedy exploration (less/none needed with noisy nets)
        if not use_noisy:
            epsilon = max(epsilon_final, epsilon_start - frame_idx / EPSILON_DECAY_LAST_FRAME)
        
        # Reset noise periodically if using noisy networks
        if use_noisy and frame_idx % NOISE_RESET_FRAMES == 0:
            net.reset_noise()
            
        reward = agent.play_step(net, device, epsilon, use_noisy)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            
            # Log SNR if using noisy networks
            if use_noisy and frame_idx % 1000 == 0:
                snr_values = net.noisy_layers_sigma_snr()
                for i, snr in enumerate(snr_values):
                    writer.add_scalar(f"snr/layer_{i}", snr, frame_idx)
                    
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), DEFAULT_MODEL_DIR + "/" + args.env.replace("/", "_") + "_" + param_id + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            # Also reset target network noise if using noisy nets
            if use_noisy:
                tgt_net.reset_noise()

        optimizer.zero_grad()
        batch, indices, weights = buffer.sample(BATCH_SIZE)
        loss_t, td_errors = calc_loss(batch, indices, weights, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()

        # Update priorities
        priorities = np.clip(td_errors + 1e-5, 0, 10)  
        buffer.update_priorities(indices, priorities)
    writer.close()