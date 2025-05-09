#!/usr/bin/env python3
import gymnasium as gym
import argparse
import numpy as np
import typing as tt
import os
import sys
import re

import torch

from lib import wrappers
from lib.noisy_dqn_model import NoisyDQN, LargeNoisyDQN

import collections

DEFAULT_ENV_NAME = "RiverraidNoFrameskip-v4"
DEFAULT_MODEL = "models/RiverraidNoFrameskip-v4_LargeNoisyDQN_ScreenSize=84_Sticky=0.1_RandStart=20_FrmSkip=2_RplSize=450000_LR=0.0001-best_eval_-86.dat"
DEFAULT_RECORD_DIR = "recordings"

# Ultimate fallback defaults (aligned with dqn_riverraid.py defaults)
SCRIPT_DEFAULT_FRAMESKIP = 2
SCRIPT_DEFAULT_STICKY_ACTION_PROB = 0.0
SCRIPT_DEFAULT_RANDOM_START_FRAMES = 30 # dqn_riverraid.py uses 30
SCRIPT_DEFAULT_USE_ACTION_MASK = False
SCRIPT_DEFAULT_FRAME_SIZE = 84
USE_LARGE_MODEL = False

def parse_model_filename_for_defaults(model_path: str) -> dict:
    """Parses known hyperparameters from the model filename."""
    defaults = {
        "use_large_model": USE_LARGE_MODEL,
        "frameskip": SCRIPT_DEFAULT_FRAMESKIP,
        "sticky": SCRIPT_DEFAULT_STICKY_ACTION_PROB,
        "random_starts": SCRIPT_DEFAULT_RANDOM_START_FRAMES,
        "use_action_mask": SCRIPT_DEFAULT_USE_ACTION_MASK,
        "screen_size": SCRIPT_DEFAULT_FRAME_SIZE
    }
    if not model_path:
        return defaults

    model_filename = os.path.basename(model_path)
    
    fs_match = re.search(r"_FrmSkip=(\d+)", model_filename)
    if fs_match:
        defaults["frameskip"] = int(fs_match.group(1))

    sticky_match = re.search(r"_Sticky=([\d\.]+)", model_filename)
    if sticky_match:
        defaults["sticky"] = float(sticky_match.group(1))

    rs_match = re.search(r"_RandStart=(\d+)", model_filename)
    if rs_match:
        defaults["random_starts"] = int(rs_match.group(1))

    if re.search(r"_UseActionMask", model_filename):
        defaults["use_action_mask"] = True

    if re.search(r"_LargeNoisyDQN", model_filename):
        defaults["use_large_model"] = True

    screen_size_match = re.search(r"_ScreenSize=(\d+)", model_filename)
    if screen_size_match:
        defaults["screen_size"] = int(screen_size_match.group(1))
        
    return defaults

if __name__ == "__main__":
    model_path_for_defaults = DEFAULT_MODEL 
    # Check if -m or --model is in sys.argv to override
    for i, arg_val in enumerate(sys.argv):
        if arg_val == '-m' or arg_val == '--model':
            if i + 1 < len(sys.argv):
                model_path_for_defaults = sys.argv[i+1]
            break
    
    parsed_defaults = parse_model_filename_for_defaults(model_path_for_defaults)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Path to the model file (default: {DEFAULT_MODEL})")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help=f"Environment name to use (default: {DEFAULT_ENV_NAME})")
    parser.add_argument("-r", "--record", default=DEFAULT_RECORD_DIR,
                        help=f"Directory to store recordings (default: {DEFAULT_RECORD_DIR})")
    
    parser.add_argument("--frameskip", type=int, default=parsed_defaults["frameskip"],
                        help=f"Number of frames to skip (default: from model filename, fallback {SCRIPT_DEFAULT_FRAMESKIP})")
    parser.add_argument("--sticky", type=float, default=parsed_defaults["sticky"],
                        help=f"Probability of sticky actions (default: from model filename, fallback {SCRIPT_DEFAULT_STICKY_ACTION_PROB})")
    parser.add_argument("--random-starts", type=int, default=parsed_defaults["random_starts"],
                        help=f"Max random no-op actions at episode start (default: from model filename, fallback {SCRIPT_DEFAULT_RANDOM_START_FRAMES})")
    parser.add_argument("--use-action-mask", default=parsed_defaults["use_action_mask"],
                        action=argparse.BooleanOptionalAction, # Requires Python 3.9+
                        help="Use reduced action space (default: from model filename, fallback False)")
    parser.add_argument("--screen-size", type=int, default=parsed_defaults["screen_size"],
                        help=f"Screen size (default: from model filename, fallback {SCRIPT_DEFAULT_FRAME_SIZE})")
    parser.add_argument("--use-large-model", default=parsed_defaults["use_large_model"], 
                        action=argparse.BooleanOptionalAction,
                        help="Use large model")
    
    args = parser.parse_args()

    print("\nEffective parameters for playback:")
    print(f"  Model: {args.model}")
    print(f"  Frameskip: {args.frameskip}")
    print(f"  Sticky Action Probability: {args.sticky}")
    print(f"  Random Start Frames: {args.random_starts}")
    print(f"  Use Action Mask: {args.use_action_mask}")
    print(f"  Use Large Model: {args.use_large_model}\n")

    record_dir = args.record + "/" + os.path.basename(args.model).split(".")[0]
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    env = wrappers.make_env(
        args.env, 
        frameskip=args.frameskip,
        sticky_action_prob=args.sticky,
        random_start_frames=args.random_starts,
        terminal_reward=-100.0,  
        fuel_reward=0.1,        
        screen_size=args.screen_size,         
        render_mode="rgb_array",
        use_action_mask=args.use_action_mask,
    )
    env = gym.wrappers.RecordVideo(env, video_folder=record_dir)

    if args.use_large_model:
        net = LargeNoisyDQN(env.observation_space.shape, env.action_space.n)
    else:
        net = NoisyDQN(env.observation_space.shape, env.action_space.n)
    
    try:
        state_dict = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure your PyTorch version is compatible with 'weights_only=True' or remove it if using an older version and trust the source.")
        sys.exit(1)
        
    net.load_state_dict(state_dict)
    net.eval()
    if hasattr(net, 'reset_noise'): # Check if it's NoisyDQN
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
