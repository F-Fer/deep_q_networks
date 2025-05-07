#!/usr/bin/env python3
import gymnasium as gym
import argparse
import numpy as np
import typing as tt
import os

import torch

from lib import wrappers
from lib.noisy_dqn_model import NoisyDQN

import collections

DEFAULT_ENV_NAME = "RiverraidNoFrameskip-v4"
DEFAULT_MODEL = "models/RiverraidNoFrameskip-v4_NoisyDQN_ScreenSize=84_Sticky=0.2_RandStart=30_FrmSkip=2_RplSize=500000_LR=0.0001-best_-93.dat"
DEFAULT_RECORD_DIR = "recordings"

# Defaults from dqn_riverraid.py for new arguments
DEFAULT_FRAMESKIP = 2
DEFAULT_STICKY_ACTION_PROB = 0.0
DEFAULT_RANDOM_START_FRAMES = 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", default=DEFAULT_RECORD_DIR)
    parser.add_argument("--frameskip", type=int, default=DEFAULT_FRAMESKIP,
                        help=f"Number of frames to skip (default: {DEFAULT_FRAMESKIP})")
    parser.add_argument("--sticky", type=float, default=DEFAULT_STICKY_ACTION_PROB,
                        help=f"Probability of sticky actions (default: {DEFAULT_STICKY_ACTION_PROB})")
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_START_FRAMES,
                        help=f"Max random no-op actions at episode start (default: {DEFAULT_RANDOM_START_FRAMES})")
    args = parser.parse_args()

    record_dir = args.record + "/" + DEFAULT_MODEL.split("/")[-1].split(".")[0]
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    env = wrappers.make_env(
        args.env, 
        frameskip=args.frameskip,
        sticky_action_prob=args.sticky,
        random_start_frames=args.random_starts,
        terminal_reward=-100.0,
        fuel_reward=0.1,
        screen_size=84,
        render_mode="rgb_array"
    )
    env = gym.wrappers.RecordVideo(env, video_folder=record_dir)
    net = NoisyDQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    net.reset_noise()

    state, _ = env.reset()
    total_reward = 0.0
    c: tt.Dict[int, int] = collections.Counter()

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()
