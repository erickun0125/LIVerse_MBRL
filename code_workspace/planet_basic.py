import argparse
import functools
import os
import pathlib
import sys
import random
from PIL import Image 
import torchvision.transforms as T
import numpy as np
import ruamel.yaml as yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Set MUJOCO environment variable
os.environ["MUJOCO_GL"] = "egl"

sys.path.append(str(pathlib.Path(__file__).parent))

# Import modules
from liv import load_liv
import clip
import models
import tools
import envs.wrappers as wrappers
from parallel import Damy

import torch
import torch.nn.functional as F

to_np = lambda x: x.detach().cpu().numpy()


class MPCPlanner:
    """Model-predictive control planner with cross-entropy method and learned transition model."""
    
    def __init__(self, transition_model, reward_function, planning_horizon, optimization_iters, 
                 candidates, top_candidates, action_size, min_action, max_action, device, reward_form="similarity"):
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.planning_horizon = planning_horizon
        self.optimization_iters = optimization_iters
        self.candidates = candidates
        self.top_candidates = top_candidates
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.device = device
        self.reward_form = reward_form
    
    def plan(self, belief, state, target_embedding):
        """Plan action sequence using CEM and return first action."""
        B, H, Z = belief.size(0), belief.size(1), state["stoch"].size(1)
        
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=self.device)
        action_std_dev = torch.ones(self.planning_horizon, B, 1, self.action_size, device=self.device)
        
        # Expand belief and state for batch processing
        belief_expanded = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H)
        # Need to expand all state components
        state_expanded = {k: v.unsqueeze(dim=1).expand(B, self.candidates, v.size(1), *v.shape[2:]).reshape(-1, v.size(1), *v.shape[2:]) 
                          for k, v in state.items()}
        
        # CEM optimization loop
        for _ in range(self.optimization_iters):
            # Sample action sequences from the current distribution
            actions = (action_mean + action_std_dev * torch.randn(
                self.planning_horizon, B, self.candidates, self.action_size, device=self.device
            )).view(self.planning_horizon, B * self.candidates, self.action_size)
            
            # Clip action range
            actions = torch.clamp(actions, min=self.min_action, max=self.max_action)
            
            # Initialize returns
            returns = torch.zeros(B * self.candidates, device=self.device)
            
            # Copy initial states
            curr_belief = belief_expanded.clone()
            curr_state = {k: v.clone() for k, v in state_expanded.items()}
            
            # Calculate initial similarity
            feat = self.transition_model.get_feat(curr_state)
            prev_similarities = self.reward_function(feat, curr_state, actions[0], target_embedding)
            
            # Rollout imagined trajectory and calculate returns
            for t in range(self.planning_horizon):
                # Get action for this timestep
                action = actions[t]
                
                # Imagine next state
                next_beliefs, _, _, _, next_states, _, _ = self.transition_model(
                    curr_state, action.unsqueeze(0), curr_belief.unsqueeze(0), None, None)
                
                # Get belief and state
                curr_belief = next_beliefs.squeeze(0)
                curr_state = {k: v.squeeze(0) for k, v in next_states.items()}
                
                # Calculate reward
                feat = self.transition_model.get_feat(curr_state)
                current_similarities = self.reward_function(feat, curr_state, action, target_embedding)
                
                # Calculate reward (difference-based or similarity itself)
                if self.reward_form == "difference":
                    reward = current_similarities - prev_similarities
                else:  # similarity
                    reward = current_similarities
                
                # Store current similarity as previous similarity for next step
                prev_similarities = current_similarities.clone()
                
                # Accumulate returns
                returns += reward
            
            # Reshape returns to (B, candidates)
            returns = returns.view(B, self.candidates)
            
            # Get top k action sequences
            _, topk = returns.topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, device=self.device).unsqueeze(dim=1)
            
            # Get best action sequences
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
            
            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(dim=2, keepdim=True)
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)
        
        # Return first action mean
        return action_mean[0].squeeze(dim=1)


def make_env(config, mode, id, liv=None):
    """Create environment based on configuration."""
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc
        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "ML1":
        print("Loading Meta-World...")
        from metaworld_wrapper import MetaWorldEnvWrapper
        env = MetaWorldEnvWrapper(task_name=task, liv=liv, seed=config.seed + id, mode=mode)
    elif suite == "atari":
        import envs.atari as atari
        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    
    env = wrappers.TimeLimit(env, config.time_limit)
    #env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def update_belief_and_act(world_model, planner, belief, posterior_state, action, observation, is_first, target_embedding):
    """Update belief and state with new observation, then plan action."""
    # Preprocess observation (important!)
    obs_processed = world_model.preprocess(observation)
    
    # Generate embedding from preprocessed observation
    embed = world_model.encoder(obs_processed).unsqueeze(dim=0)
    
    # Update state
    belief, _, _, _, posterior_state, _, _ = world_model.dynamics(
        posterior_state, action.unsqueeze(dim=0), belief, embed, is_first)
    
    # Remove time dimension
    belief, posterior_state = belief.squeeze(dim=0), {k: v.squeeze(dim=0) for k, v in posterior_state.items()}
    
    # Plan action
    action = planner.plan(belief, posterior_state, target_embedding)
    
    return belief, posterior_state, action


def save_video(frames, filename, fps=30):
    """Save frames as MP4 video."""
    frames = np.array(frames)
    if frames.shape[-1] == 3:  # RGB format
        fig, ax = plt.subplots(figsize=(frames.shape[2]/100, frames.shape[1]/100))
        plt.tight_layout()
        plt.axis('off')
        
        def animate(i):
            ax.clear()
            ax.imshow(frames[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            plt.tight_layout()
            return []
        
        anim = FuncAnimation(fig, animate, frames=len(frames))
        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()
    else:
        raise ValueError("Frames should be in RGB format")


def plot_similarity(timesteps, similarities, diff_rewards, filename=None):
    """Plot similarity scores and difference rewards over time and save to file."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Similarity graph (left y-axis)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Similarity Score', color='tab:blue')
    ax1.plot(timesteps, similarities, 'b-', label='Similarity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Difference-based reward graph (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Difference Reward', color='tab:red')
    # Visualize difference-based rewards excluding the first timestep (no previous value for first timestep)
    ax2.plot(timesteps[1:], diff_rewards[1:], 'r-', label='Difference')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Title and legend
    plt.title('Similarity Score and Difference Reward over Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()
    
    return fig


def get_next_available_dir(base_dir):
    """Return automatically numbered new directory path if directory already exists"""
    # Analyze base path
    base_path = pathlib.Path(base_dir)
    parent_dir = base_path.parent
    base_name = base_path.name
    
    # Extract number (e.g., separate "planet_results" and "1" from "planet_results1")
    import re
    match = re.match(r"(.*?)(\d*)$", base_name)
    
    if match:
        prefix = match.group(1)
        # Start from the number if it exists, otherwise start from 1
        start_num = int(match.group(2)) if match.group(2) else 1
    else:
        # If no number pattern, use entire name as prefix and start from 1
        prefix = base_name
        start_num = 1
    
    # Check existing directories and find next available number
    current_num = start_num
    while True:
        # Create folder path with current number
        current_dir = parent_dir / f"{prefix}{current_num}"
        
        # Return this path if the folder doesn't exist
        if not current_dir.exists():
            return str(current_dir)
        
        # Increment number if it exists
        current_num += 1

def main(config):
    """Run MPC planning with pre-trained Dreamer world model."""
    # Set up device
    if config.device and torch.cuda.is_available():
        device = torch.device(config.device)
    elif torch.cuda.is_available() and not config.disable_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Set random seeds
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    
    # Set up directories
    logdir = pathlib.Path(config.logdir).expanduser()

    # Auto-numbering for results_dir
    if hasattr(config, 'results_dir') and config.results_dir:
        # Use as specified if user directly specified
        results_dir = pathlib.Path(config.results_dir).expanduser()
        print(f"Using user specified results directory: {results_dir}")
    else:
        # Apply auto-numbering to default directory
        base_results_dir = "./planet_results/planet_results1"
        results_dir = pathlib.Path(get_next_available_dir(base_results_dir))
        print(f"Using auto-numbered results directory: {results_dir}")

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    print("Loading VLM...")
    liv = load_liv()
    liv.eval()
    
    # Process goal_image
    goal_image_base_path = "logdir/planet_test"
    if hasattr(config, 'goal_image') and config.goal_image:
        goal_image_filename = f"{config.goal_image}.png"
    else:
        goal_image_filename = "success_frame_button.png"
    
    goal_image_path = os.path.join(goal_image_base_path, goal_image_filename)
    
    if os.path.exists(goal_image_path):
        print(f"Using '{goal_image_path}' as goal image.")
        # Load and transform image
        transform = T.Compose([T.ToTensor()])
        goal_pil_image = Image.open(goal_image_path).convert('RGB')
        goal_tensor = transform(goal_pil_image).unsqueeze(0).to('cuda:0')
        
        # Generate image embedding
        with torch.no_grad():
            target_text_embedding = liv(input=goal_tensor, modality="vision")
        print("Generated image-based goal embedding.")
    else:
        print(f"Warning: Image '{goal_image_path}' not found. Using text-based embedding.")
        # Use existing text embedding
        text = clip.tokenize([config.text_prompt]).to('cuda:0')
        with torch.no_grad():
            target_text_embedding = liv(input=text, modality="text")
        print(f"Generated text-based goal embedding ('{config.text_prompt}').")
    
    print(f"Reward form: {config.reward_form}")
    
    print("Creating environment...")
    env = make_env(config, "eval", 0, liv)
    #env = Damy(env)
    
    # Set num_actions in config (missing from the original code)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"Action space: {acts}, num_actions: {config.num_actions}")
    
    print("Loading Dreamer model...")
    from dreamer import Dreamer  # Import Dreamer class
    
    # Initialize dummy dataset (not used for evaluation)
    dummy_dataset = iter([{
        'image': torch.zeros(1, config.batch_length, 64, 64, 3, device=device),
        'action': torch.zeros(1, config.batch_length, env.action_space.shape[0], device=device),
        'reward': torch.zeros(1, config.batch_length, device=device),
        'discount': torch.zeros(1, config.batch_length, 1, device=device),
        'is_first': torch.zeros(1, config.batch_length, 1, dtype=torch.int8, device=device),
        'is_terminal': torch.zeros(1, config.batch_length, 1, dtype=torch.int8, device=device),
    }])
    
    # Initialize dummy logger
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def scalar(self, *args): pass
        def video(self, *args): pass
        def write(self, *args): pass
    dummy_logger = DummyLogger()
    
    # Initialize Dreamer
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        dummy_logger,
        dummy_dataset,
        target_text_embedding,
    ).to(device)
    
    # Load model from checkpoint
    checkpoint_file = "latest_button.pt"
    if hasattr(config, 'checkpoint_file') and config.checkpoint_file:
        checkpoint_file = config.checkpoint_file
    
    checkpoint_path = logdir / checkpoint_file
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print("Model loaded successfully")
    
    # Extract world model for planning
    world_model = agent._wm
    world_model.eval()
    
    # Define reward function
    reward_fn = lambda f, s, a, target: F.cosine_similarity(
        world_model.heads["reward"](f), target, dim=-1
    )
    
    # Create MPC planner
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
        
        # Reset environment
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
        
        # Record similarity and difference-based rewards by timestep
        timesteps = []
        similarities = []
        diff_rewards = []
        prev_similarity = 0.0
        
        # Run episode
        with torch.no_grad():
            pbar = tqdm(range(config.max_episode_length // config.action_repeat))
            for t in pbar:
                # Render environment
                if config.render:
                    frame = env.render()
                    video_frames.append(frame)
                
                # Update belief and plan action
                belief, posterior_state, action = update_belief_and_act(
                    world_model,  # 전체 월드 모델 전달
                    planner,
                    belief,
                    posterior_state,
                    action,
                    observation,  # 여기서는 원본 관측값을 전달
                    torch.tensor(observation["is_first"], device=device),
                    target_text_embedding
                )
                
                # Calculate current similarity
                feat = world_model.dynamics.get_feat(posterior_state)
                current_similarity = F.cosine_similarity(
                    world_model.heads["reward"](feat), target_text_embedding, dim=-1
                ).item()
                
                # Calculate difference-based reward
                diff_reward = current_similarity - prev_similarity
                prev_similarity = current_similarity
                
                # Record data
                timesteps.append(t)
                similarities.append(current_similarity)
                diff_rewards.append(diff_reward)
                
                # Clip action range
                action_np = to_np(action.cpu())
                action_np = np.clip(action_np, -1.0, 1.0)
                
                # Step environment
                observation, reward, done, _ = env.step(action_np[0])
                total_reward += reward
                
                # Update progress bar
                if config.reward_form == "difference":
                    pbar.set_description(f"Reward: {total_reward:.2f}, Similarity: {current_similarity:.4f}, Diff: {diff_reward:.4f}")
                else:
                    pbar.set_description(f"Reward: {total_reward:.2f}, Similarity: {current_similarity:.4f}")
                
                if done:
                    break
        
        print(f"Episode {episode} completed with reward {total_reward:.2f}")
        
        # Save graph
        plot_filename = results_dir / f"similarity_plot_episode_{episode}_{config.reward_form}.png"
        plot_similarity(timesteps, similarities, diff_rewards, plot_filename)
        print(f"Similarity plot saved to {plot_filename}")
        
        # Save video
        if video_frames and config.save_video:
            video_path = results_dir / f"planet_episode_{episode}_{config.reward_form}.mp4"
            print(f"Saving video to {video_path}")
            save_video(video_frames, str(video_path))
    
    env.close()
    print("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaNet-style MPC with Dreamer's World Model")
    
    # Add configs argument similar to dreamer.py
    parser.add_argument("--configs", nargs="+", help="Configuration files to load")
    
    # Parse just the configs argument first
    configs_args, remaining = parser.parse_known_args()
    
    # Load YAML configuration
    configs_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.safe_load(configs_path.read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    # Start with default configuration
    defaults = {}
    name_list = ["defaults"]
    if configs_args.configs:
        name_list.extend(configs_args.configs)
    
    for name in name_list:
        if name in configs:
            recursive_update(defaults, configs[name])
        else:
            print(f"Warning: Config '{name}' not found in configs.yaml")
    
    # Add PlaNet-specific arguments that aren't in the defaults
    parser = argparse.ArgumentParser(description="PlaNet-style MPC with Dreamer's World Model")
    
    # Add PlaNet-specific arguments
    planet_specific = {
        "results_dir": "",  # 자동 넘버링 처리됨
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
        "goal_image": "",  # Newly added: goal image file name (without extension)
        "checkpoint_file": "latest_button.pt",  # Newly added: checkpoint file name
        # New reward form related settings
        "reward_form": "similarity"  # Reward form: similarity or difference
    }
    
    # Update defaults with PlaNet-specific defaults if not already present
    for key, value in planet_specific.items():
        if key not in defaults:
            defaults[key] = value
    
    # Create arguments from defaults
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    # Parse remaining arguments
    config = parser.parse_args(remaining)
    
    # Make sure rendering and video saving are enabled
    config.render = True
    config.save_video = True
    
    # Run main function
    main(config)