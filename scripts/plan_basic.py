"""Single-goal MPC planning with LIVerse world model."""
import argparse
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "third_party"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "third_party" / "liv" / "models"))

from liv import load_liv
import clip

from liverse.agents.dreamer import Dreamer
from liverse.envs import make_env
from liverse.parallel import Damy
from liverse.planning.planner import MPCPlanner, update_belief_and_act
from liverse.utils.math import to_np
from liverse.utils.seed import set_seed_everywhere, enable_deterministic_run
from liverse.utils.config import args_type, recursive_update
from liverse.utils.logger import DummyLogger
from liverse.utils.visualization import save_video, plot_similarity, get_next_available_dir


def main(config):
    """Run MPC planning with pre-trained Dreamer world model."""
    if config.device and torch.cuda.is_available():
        device = torch.device(config.device)
    elif torch.cuda.is_available() and not config.disable_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    set_seed_everywhere(config.seed)
    if config.deterministic_run:
        enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()

    if hasattr(config, 'results_dir') and config.results_dir:
        results_dir = pathlib.Path(config.results_dir).expanduser()
    else:
        base_results_dir = "./planet_results/planet_results1"
        results_dir = pathlib.Path(get_next_available_dir(base_results_dir))

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    print("Loading VLM...")
    liv = load_liv()
    liv.eval()

    goal_image_base_path = "logdir/planet_test"
    if hasattr(config, 'goal_image') and config.goal_image:
        goal_image_filename = f"{config.goal_image}.png"
    else:
        goal_image_filename = "success_frame_button.png"

    goal_image_path = os.path.join(goal_image_base_path, goal_image_filename)

    if os.path.exists(goal_image_path):
        print(f"Using '{goal_image_path}' as goal image.")
        transform = T.Compose([T.ToTensor()])
        goal_pil_image = Image.open(goal_image_path).convert('RGB')
        goal_tensor = transform(goal_pil_image).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            target_text_embedding = liv(input=goal_tensor, modality="vision")
    else:
        print(f"Warning: Image '{goal_image_path}' not found. Using text-based embedding.")
        text = clip.tokenize([config.text_prompt]).to('cuda:0')
        with torch.no_grad():
            target_text_embedding = liv(input=text, modality="text")

    print(f"Reward form: {config.reward_form}")

    print("Creating environment...")
    env = make_env(config, "eval", 0, liv)

    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    print("Loading Dreamer model...")

    dummy_dataset = iter([{
        'image': torch.zeros(1, config.batch_length, 64, 64, 3, device=device),
        'action': torch.zeros(1, config.batch_length, env.action_space.shape[0], device=device),
        'reward': torch.zeros(1, config.batch_length, device=device),
        'discount': torch.zeros(1, config.batch_length, 1, device=device),
        'is_first': torch.zeros(1, config.batch_length, 1, dtype=torch.int8, device=device),
        'is_terminal': torch.zeros(1, config.batch_length, 1, dtype=torch.int8, device=device),
    }])

    dummy_logger = DummyLogger()

    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        dummy_logger,
        dummy_dataset,
        target_text_embedding,
    ).to(device)

    checkpoint_file = "latest_button.pt"
    if hasattr(config, 'checkpoint_file') and config.checkpoint_file:
        checkpoint_file = config.checkpoint_file

    checkpoint_path = logdir / checkpoint_file
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print("Model loaded successfully")

    world_model = agent._wm
    world_model.eval()

    reward_fn = lambda f, s, a, target: F.cosine_similarity(
        world_model.heads["reward"](f), target, dim=-1
    )

    planner = MPCPlanner(
        transition_model=world_model.dynamics,
        reward_function=reward_fn,
        planning_horizon=config.planning_horizon,
        optimization_iters=config.optimization_iters,
        candidates=config.candidates,
        top_candidates=config.top_candidates,
        action_size=env.action_space.shape[0],
        min_action=-1.0,
        max_action=1.0,
        device=device,
        reward_form=config.reward_form
    )

    print(f"Running {config.eval_episodes} evaluation episodes...")

    for episode in range(1, config.eval_episodes + 1):
        print(f"Episode {episode}/{config.eval_episodes}")

        observation = env.reset()
        belief = torch.zeros(1, config.dyn_hidden, device=device)
        posterior_state = {
            'stoch': torch.zeros(1, world_model.dynamics._stoch, device=device),
            'deter': torch.zeros(1, world_model.dynamics._deter, device=device),
        }
        if world_model.dynamics._discrete:
            posterior_state['logit'] = torch.zeros(1, world_model.dynamics._stoch, world_model.dynamics._discrete, device=device)
        else:
            posterior_state['mean'] = torch.zeros(1, world_model.dynamics._stoch, device=device)
            posterior_state['std'] = torch.zeros(1, world_model.dynamics._stoch, device=device)

        action = torch.zeros(1, env.action_space.shape[0], device=device)
        done = False
        total_reward = 0
        video_frames = []

        timesteps = []
        similarities = []
        diff_rewards = []
        prev_similarity = 0.0

        with torch.no_grad():
            pbar = tqdm(range(config.max_episode_length // config.action_repeat))
            for t in pbar:
                if config.render:
                    frame = env.render()
                    video_frames.append(frame)

                belief, posterior_state, action = update_belief_and_act(
                    world_model, planner, belief, posterior_state, action,
                    observation, torch.tensor(observation["is_first"], device=device),
                    target_text_embedding
                )

                feat = world_model.dynamics.get_feat(posterior_state)
                current_similarity = F.cosine_similarity(
                    world_model.heads["reward"](feat), target_text_embedding, dim=-1
                ).item()

                diff_reward = current_similarity - prev_similarity
                prev_similarity = current_similarity

                timesteps.append(t)
                similarities.append(current_similarity)
                diff_rewards.append(diff_reward)

                action_np = to_np(action.cpu())
                action_np = np.clip(action_np, -1.0, 1.0)

                observation, reward, done, _ = env.step(action_np[0])
                total_reward += reward

                pbar.set_description(f"Reward: {total_reward:.2f}, Similarity: {current_similarity:.4f}")

                if done:
                    break

        print(f"Episode {episode} completed with reward {total_reward:.2f}")

        plot_filename = results_dir / f"similarity_plot_episode_{episode}_{config.reward_form}.png"
        plot_similarity(timesteps, similarities, diff_rewards, filename=plot_filename)

        if video_frames and config.save_video:
            video_path = results_dir / f"planet_episode_{episode}_{config.reward_form}.mp4"
            save_video(video_frames, str(video_path))

    env.close()
    print("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaNet-style MPC with Dreamer's World Model")
    parser.add_argument("--configs", nargs="+")
    configs_args, remaining = parser.parse_known_args()

    configs_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    configs = yaml.safe_load((configs_dir / "dreamer.yaml").read_text())

    defaults = {}
    name_list = ["defaults"]
    if configs_args.configs:
        name_list.extend(configs_args.configs)
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])

    planet_specific = {
        "results_dir": "",
        "planning_horizon": 12,
        "optimization_iters": 10,
        "candidates": 1000,
        "top_candidates": 50,
        "eval_episodes": 1,
        "text_prompt": "Robot arm presses a button.",
        "render": True,
        "save_video": True,
        "max_episode_length": 100,
        "disable_cuda": False,
        "goal_image": "",
        "checkpoint_file": "latest_button.pt",
        "reward_form": "similarity",
    }
    for key, value in planet_specific.items():
        if key not in defaults:
            defaults[key] = value

    parser = argparse.ArgumentParser(description="PlaNet-style MPC with Dreamer's World Model")
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    config = parser.parse_args(remaining)
    config.render = True
    config.save_video = True
    main(config)
