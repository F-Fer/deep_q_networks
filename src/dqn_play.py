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
DEFAULT_MODEL = "/home/finnf/dev/deep_q_networks/models/RiverraidNoFrameskip-v4_NoisyDQN_ClipRwd_PrioReplay_WithFrmSkip_NoRptAction_FuelReward_NegTrmlRwd=-50_NoopMax=30_ActionMask_FrmSkip=1_RplSize=250000_LR=0.00025_EpsFinal=0.0_EpsDecayLastFrame=250000_RedStartSize=50000_Gamma=0.99_BatchSize=32-best_-29.dat"
DEFAULT_RECORD_DIR = "recordings"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", default=DEFAULT_RECORD_DIR)
    args = parser.parse_args()

    record_dir = args.record + "/" + DEFAULT_MODEL.split("/")[-1].split(".")[0]
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    env = wrappers.make_env(args.env, render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.RecordVideo(env, video_folder=record_dir)
    net = NoisyDQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state)

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
