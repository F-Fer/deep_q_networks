import torch 
import numpy as np
import os
import gym
from tqdm import tqdm

from src.lib.noisy_dqn_model import NoisyDQN
from src.lib.wrappers import make_env

MODEL_DIR = "/home/finnf/dev/deep_q_networks/models" 
NUM_TEST_EPISODES = 10
FRAMESKIP = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(env, model_path):
    model = NoisyDQN(env.observation_space.shape, env.action_space.n)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

def test_model(model, env, num_episodes=NUM_TEST_EPISODES):
    total_reward = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_v = torch.tensor(np.expand_dims(state, 0))
            q_vals = model(state_v).data.numpy()[0]
            action = int(np.argmax(q_vals))
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        total_reward.append(episode_reward)
    avg_reward = np.mean(total_reward)
    return avg_reward

def main():
    model_scores = {}
    env = make_env("RiverraidNoFrameskip-v4", frameskip=FRAMESKIP)
    for model_path in tqdm(os.listdir(MODEL_DIR)):
        model = load_model(env, os.path.join(MODEL_DIR, model_path))
        avg_reward = test_model(model, env)
        model_scores[model_path] = avg_reward
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    print("Model Scores:")
    for model, score in sorted_models:
        print(f"{model}: {score:.2f}")

if __name__ == "__main__":
    main()