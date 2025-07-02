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
import glob

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
        
        # 차이 기반 보상을 위한 변수 추가
        self.prev_similarities = None
    
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
            
            # 초기 유사도 계산
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
                
                # 현재 유사도 계산
                feat = self.transition_model.get_feat(curr_state)
                current_similarities = self.reward_function(feat, curr_state, action, target_embedding)
                
                # 보상 계산 (차이 기반 또는 유사도 자체)
                if self.reward_form == "difference":
                    reward = current_similarities - prev_similarities
                else:  # similarity
                    reward = current_similarities
                
                # 현재 유사도를 다음 단계의 이전 유사도로 저장
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
    # 관측값 전처리 (중요!)
    obs_processed = world_model.preprocess(observation)
    
    # 전처리된 관측값으로 임베딩 생성
    embed = world_model.encoder(obs_processed).unsqueeze(dim=0)
    
    # 상태 업데이트
    belief, _, _, _, posterior_state, _, _ = world_model.dynamics(
        posterior_state, action.unsqueeze(dim=0), belief, embed, is_first)
    
    # 시간 차원 제거
    belief, posterior_state = belief.squeeze(dim=0), {k: v.squeeze(dim=0) for k, v in posterior_state.items()}
    
    # 액션 계획
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


def plot_similarity(timesteps, similarities, diff_rewards, subtask_changes=None, filename=None):
    """Plot similarity scores over time and save to file."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 유사도 그래프 (왼쪽 y축)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Similarity Score', color='tab:blue')
    ax1.plot(timesteps, similarities, 'b-', label='Similarity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # 차이 기반 보상 그래프 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Difference Reward', color='tab:red')
    ax2.plot(timesteps[1:], diff_rewards[1:], 'r-', label='Difference')  # 첫 번째 타임스텝 제외
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # 서브태스크 변경 지점 표시
    if subtask_changes:
        for t, subtask_id in subtask_changes:
            ax1.axvline(x=t, color='g', linestyle='--', alpha=0.5)
            ax1.text(t, max(similarities)*0.9, f'Task {subtask_id}', 
                     rotation=90, verticalalignment='top')
    
    # 제목과 범례
    plt.title('Similarity Score and Difference Reward over Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()
    
    return fig


def load_subtasks(subtask_dir, liv, device):
    """서브태스크 이미지들을 로드하고 임베딩 생성."""
    print(f"Loading subtasks from {subtask_dir}...")
    
    # 이미지 변환기
    transform = T.Compose([T.ToTensor()])
    
    # 이미지 파일 찾기 (숫자 순서로 정렬)
    image_files = sorted(glob.glob(os.path.join(subtask_dir, "*.png")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(subtask_dir, "*.jpg")))
    
    if not image_files:
        raise ValueError(f"No image files found in {subtask_dir}")
    
    # 임베딩 생성
    subtask_embeddings = []
    for i, img_path in enumerate(image_files):
        print(f"Loading subtask {i+1}: {os.path.basename(img_path)}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = liv(input=img_tensor, modality="vision")
        subtask_embeddings.append(embedding)
    
    print(f"Loaded {len(subtask_embeddings)} subtasks")
    return subtask_embeddings


def get_next_available_dir(base_dir):
    """디렉토리가 이미 존재하는 경우 자동으로 넘버링된 새 디렉토리 경로 반환"""
    if not os.path.exists(base_dir):
        return base_dir
    
    # 기본 디렉토리 이름을 추출
    base_path = pathlib.Path(base_dir)
    parent_dir = base_path.parent
    base_name = base_path.name
    
    # 숫자 패턴 추출 (예: planet_stepwise_results1에서 1 추출)
    import re
    pattern = re.compile(r'(.+?)(\d*)$')
    match = pattern.match(base_name)
    
    if match:
        prefix = match.group(1)
        # 숫자가 있으면 가져오고, 없으면 1로 시작
        num = int(match.group(2)) if match.group(2) else 1
        
        # 이미 존재하는 경우 다음 번호 찾기
        while True:
            next_name = f"{prefix}{num}"
            next_path = parent_dir / next_name
            if not os.path.exists(next_path):
                return str(next_path)
            num += 1
    
    # 매칭이 안되면 그냥 숫자 추가
    i = 1
    while True:
        next_path = parent_dir / f"{base_name}{i}"
        if not os.path.exists(next_path):
            return str(next_path)
        i += 1


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
    
    # results_dir 자동 넘버링
    if hasattr(config, 'results_dir') and config.results_dir:
        base_results_dir = config.results_dir
    else:
        # 기본 결과 디렉토리 이름 - 단계적 모드인지 여부에 따라 다름
        if config.step_wise:
            base_results_dir = "./planet_results/planet_stepwise_results1"
        else:
            base_results_dir = "./planet_results/planet_results1"
    
    results_dir = pathlib.Path(get_next_available_dir(base_results_dir)).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    print("Loading VLM...")
    liv = load_liv()
    liv.eval()
    
    # 타겟 임베딩 - 일반 또는 step_wise 모드에 따라 다름
    target_embeddings = []
    
    if config.step_wise:
        print(f"Using step-wise mode with threshold {config.similarity_threshold}")
        target_embeddings = load_subtasks(config.subtask_dir, liv, device)
    else:
        goal_image_path = "logdir/planet_test/success_frame_button.png"  # 성공 프레임 경로 설정
        if goal_image_path and os.path.exists(goal_image_path):
            print(f"목표 이미지로 '{goal_image_path}'를 사용합니다.")
            # 이미지 로드 및 변환
            transform = T.Compose([T.ToTensor()])
            goal_pil_image = Image.open(goal_image_path).convert('RGB')
            goal_tensor = transform(goal_pil_image).unsqueeze(0).to('cuda:0')
            
            # 이미지 임베딩 생성
            with torch.no_grad():
                target_embedding = liv(input=goal_tensor, modality="vision")
            target_embeddings = [target_embedding]
            print("이미지 기반 목표 임베딩을 생성했습니다.")
        else:
            print(f"경고: 이미지 '{goal_image_path}'를 찾을 수 없습니다. 텍스트 기반 임베딩을 사용합니다.")
            # 기존 텍스트 임베딩 사용
            text = clip.tokenize(["Agent presses button"]).to('cuda:0')
            with torch.no_grad():
                target_embedding = liv(input=text, modality="text")
            target_embeddings = [target_embedding]
            print("텍스트 기반 목표 임베딩을 생성했습니다.")
    
    print("Creating environment...")
    env = make_env(config, "eval", 0, liv)
    
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
        target_embeddings[0],  # 첫 번째 임베딩으로 초기화
    ).to(device)
    
    # Load model from checkpoint
    print(f"Loading checkpoint from {logdir / 'latest_button.pt'}")
    checkpoint = torch.load(logdir / "latest_button.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print("Model loaded successfully")
    
    # Extract world model for planning
    world_model = agent._wm
    world_model.eval()
    
    # Define reward function (추가 매개변수로 target_embedding을 받도록 수정)
    reward_fn = lambda f, s, a, target: F.cosine_similarity(
        world_model.heads["reward"](f), target, dim=-1
    )
    
    print(f"Reward form: {config.reward_form}")
    
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

        # Step-wise 모드 변수
        current_subtask_idx = 0
        subtask_changes = []  # 서브태스크 변경 시점 기록
        plateau_counter = 0   # diff_reward 정체 상태 카운터

        # 차이 기반 보상 계산을 위한 변수
        prev_similarity = 0.0

        # 타임스텝별 유사도와 차이 기반 보상 기록
        timesteps = []
        similarities = []
        next_similarities = []
        diff_rewards = []

        # Run episode
        with torch.no_grad():
            pbar = tqdm(range(config.max_episode_length // config.action_repeat))
            for t in pbar:
                # Render environment
                if config.render:
                    frame = env.render()
                    video_frames.append(frame)
                
                # 현재 서브태스크 임베딩
                current_target = target_embeddings[current_subtask_idx]
                next_target = target_embeddings[current_subtask_idx+1] if current_subtask_idx < len(target_embeddings) - 1 else target_embeddings[-1]
                
                # Update belief and plan action
                belief, posterior_state, action = update_belief_and_act(
                    world_model,
                    planner,
                    belief,
                    posterior_state,
                    action,
                    observation,
                    torch.tensor(observation["is_first"], device=device),
                    current_target
                )
                
                # 현재 유사도 계산
                feat = world_model.dynamics.get_feat(posterior_state)
                current_similarity = F.cosine_similarity(
                    world_model.heads["reward"](feat), current_target, dim=-1
                ).item()
                
                # 다음 서브태스크에 대한 유사도 계산
                next_similarity = F.cosine_similarity(
                    world_model.heads["reward"](feat), next_target, dim=-1
                ).item() if len(target_embeddings) > 1 else 0.0
                
                # 차이 기반 보상 계산
                diff_reward = current_similarity - prev_similarity
                prev_similarity = current_similarity
                
                # Step-wise 모드에서 서브태스크 완료 확인 - plateau 감지 방식으로 변경
                if config.step_wise:
                    # diff_reward가 임계값보다 작으면 정체 상태로 카운트
                    if diff_reward < config.similarity_threshold:
                        plateau_counter += 1
                    else:
                        plateau_counter = 0  # 임계값 이상이면 카운터 리셋
                    
                    # 정해진 횟수 동안 정체 상태가 지속되면 다음 서브태스크로 전환
                    if plateau_counter >= config.plateau_patience:
                        if current_subtask_idx < len(target_embeddings) - 1:
                            current_subtask_idx += 1
                            print(f"[{t}] 서브태스크 {current_subtask_idx}로 전환! (유사도: {current_similarity:.4f}, 정체 카운트: {plateau_counter})")
                            subtask_changes.append((t, current_subtask_idx))
                            # 새 서브태스크에 대한 유사도 초기화
                            prev_similarity = 0.0
                            plateau_counter = 0  # 카운터 리셋
                
                # 기록
                timesteps.append(t)
                similarities.append(current_similarity)
                next_similarities.append(next_similarity)
                diff_rewards.append(diff_reward)
                
                # Clip action range
                action_np = to_np(action.cpu())
                action_np = np.clip(action_np, -1.0, 1.0)
                
                # Step environment
                observation, reward, done, _ = env.step(action_np[0])
                total_reward += reward
                
                # Update progress bar
                task_info = f"Task {current_subtask_idx+1}/{len(target_embeddings)}" if config.step_wise else ""
                if config.reward_form == "difference":
                    pbar.set_description(f"{task_info} Reward: {total_reward:.2f}, Similarity: {current_similarity:.4f}, Next: {next_similarity:.4f}, Diff: {diff_reward:.4f}, Plateau: {plateau_counter}")
                else:
                    pbar.set_description(f"{task_info} Reward: {total_reward:.2f}, Similarity: {current_similarity:.4f}, Next: {next_similarity:.4f}, Plateau: {plateau_counter}")
                
                if done:
                    break

        print(f"Episode {episode} completed with reward {total_reward:.2f}")

        # 그래프 저장
        step_mode = "stepwise" if config.step_wise else "single"
        plot_filename = results_dir / f"similarity_plot_ep{episode}_{config.reward_form}_{step_mode}.png"
        plot_similarity(timesteps, similarities, diff_rewards, subtask_changes, plot_filename)
        print(f"Similarity plot saved to {plot_filename}")

        # Save video
        if video_frames and config.save_video:
            video_path = results_dir / f"planet_episode_{episode}_{config.reward_form}_{step_mode}.mp4"
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
    
    # Add PlaNet-specific arguments
    planet_specific = {
        # step_wise 여부에 따라 다른 기본 디렉토리 이름 설정 (자동 넘버링됨)
        "results_dir": "",  # 비워두고 step_wise 모드에 따라 내부에서 결정
        "planning_horizon": 10,
        "optimization_iters": 10,
        "candidates": 1000,
        "top_candidates": 100,
        "eval_episodes": 1,
        "text_prompt": "Robot arm presses a button.",
        "render": True, 
        "save_video": True,
        "max_episode_length": 200,
        "disable_cuda": False,
        "reward_form": "similarity",  # 보상 형태: similarity 또는 difference
        
        # Step-wise 관련 설정 추가
        "step_wise": False,  # 서브태스크 모드 활성화 여부
        "subtask_dir": "./subtasks",  # 서브태스크 이미지 디렉토리
        "similarity_threshold": 0.0012,  # 다음 서브태스크로 넘어가는 유사도 임계값
        "plateau_patience": 5  # 다음 서브태스크로 넘어가기 전 기다리는 스텝 수
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