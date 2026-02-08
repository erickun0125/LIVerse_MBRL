"""Evaluate trained LIVerse model."""
import argparse
import os
import pathlib
import sys
from datetime import datetime

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml
import torch
import imageio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "third_party"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "third_party" / "liv" / "models"))

from liv import load_liv
import clip

from liverse.agents.dreamer import Dreamer
from liverse.envs import make_env
from liverse.utils.logger import Logger
from liverse.utils.seed import set_seed_everywhere, enable_deterministic_run
from liverse.utils.config import args_type, recursive_update


def record_episode(agent, env, text_embedding, config, save_path, max_steps=500):
    """Run one episode and save as video."""
    print(f"Starting test episode - save path: {save_path}")
    frames = []

    reset_result = env.reset()
    if callable(reset_result):
        obs = reset_result()
    else:
        obs = reset_result

    state = None
    step = 0
    total_reward = 0.0
    success = False

    if 'image' in obs:
        img = np.array(obs['image'])
        if img.shape[-1] == 3:
            frames.append(img)
        else:
            frames.append(np.transpose(img, (1, 2, 0)))

    while step < max_steps:
        policy_output, state = agent(obs, [False], state, training=False)
        action = policy_output["action"]

        step_result = env.step(action)
        if callable(step_result):
            step_output = step_result()
            if len(step_output) == 4:
                obs, reward, done, info = step_output
            elif len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
        else:
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

        if 'image' in obs:
            img = np.array(obs['image'])
            if img.shape[-1] == 3:
                frames.append(img)
            else:
                frames.append(np.transpose(img, (1, 2, 0)))

        total_reward += reward
        step += 1

        if isinstance(info, dict) and 'success' in info and info['success']:
            success = True

        if done:
            break

    print(f"Episode done - steps: {step}, reward: {total_reward}, success: {success}")

    if len(frames) > 0:
        frames = np.array(frames)
        try:
            imageio.mimsave(save_path, frames, fps=30)
            print(f"Video saved: {save_path}")
        except Exception as e:
            print(f"Error saving video: {e}")

    return {"steps": step, "total_reward": total_reward, "success": success}


def main(config):
    config.compile = False
    set_seed_everywhere(config.seed)
    if config.deterministic_run:
        enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    if not logdir.exists():
        raise FileNotFoundError(f"Log directory not found: {logdir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = logdir / f"test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)

    video_path = test_dir / f"{config.task}_test.mp4"
    checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    liv = load_liv()
    liv.eval()

    text = clip.tokenize(["Agent presses the button."]).to(config.device)
    with torch.no_grad():
        target_text_embedding = liv(input=text, modality="text")

    env = make_env(config, "eval", 0, liv)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    dummy_logger = Logger(test_dir, 0)

    agent = Dreamer(
        env.observation_space, env.action_space, config,
        dummy_logger, None, target_text_embedding,
    ).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    results = record_episode(agent, env, target_text_embedding, config, video_path)

    with open(test_dir / "results.txt", "w") as f:
        f.write(f"Task: {config.task}\n")
        f.write(f"Steps: {results['steps']}\n")
        f.write(f"Total Reward: {results['total_reward']}\n")
        f.write(f"Success: {results['success']}\n")

    print(f"Results: steps={results['steps']}, reward={results['total_reward']}, success={results['success']}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args, remaining = parser.parse_known_args()

    configs_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    configs = yaml.safe_load((configs_dir / "dreamer.yaml").read_text())

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    if args.logdir:
        config.logdir = args.logdir
    if args.task:
        config.task = args.task
    main(config)
