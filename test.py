import argparse
import functools
import os
import pathlib
import sys
import numpy as np
import ruamel.yaml as yaml
import imageio
from datetime import datetime
import torch
import torchvision.transforms as T

os.environ["MUJOCO_GL"] = "egl"
from liv import load_liv
import clip

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent / 'liv' / 'models'))

import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

to_np = lambda x: x.detach().cpu().numpy()


def make_env(config, mode, id, liv=None):
    suite, task = config.task.split("_", 1)
    if suite == "ML1":
        from metaworld_wrapper import MetaWorldEnvWrapper
        env = MetaWorldEnvWrapper(task_name=task, liv=liv, seed=config.seed + id, mode=mode)
    elif suite == "dmc":
        import envs.dmc as dmc
        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(f"Environment suite {suite} not implemented for testing")
        
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def record_episode(agent, env, text_embedding, config, save_path, max_steps=500):
    """
    한 에피소드를 실행하고 비디오로 저장하는 함수
    """
    print(f"테스트 에피소드 시작 - 저장 경로: {save_path}")
    frames = []
    
    # Damy 래퍼를 사용하는 경우 reset이 함수를 반환하므로 호출해야 함
    reset_result = env.reset()
    # 결과가 함수인 경우 호출
    if callable(reset_result):
        obs = reset_result()
    else:
        obs = reset_result
    
    state = None
    step = 0
    total_reward = 0.0
    success = False
    
    # 첫 프레임 저장
    if 'image' in obs:
        img = np.array(obs['image'])
        if img.shape[-1] == 3:  # 채널이 마지막 차원에 있는지 확인
            frames.append(img)
        else:
            # 채널 차원을 변경해야 할 경우
            frames.append(np.transpose(img, (1, 2, 0)))
    
    # 에피소드 진행
    while step < max_steps:
        policy_output, state = agent(obs, [False], state, training=False)
        action = policy_output["action"]
        
        # step 함수도 Damy 래퍼로 인해 함수를 반환할 수 있음
        step_result = env.step(action)
        if callable(step_result):
            step_output = step_result()
            # 반환값 형식에 따라 처리
            if len(step_output) == 4:  # obs, reward, done, info
                obs, reward, done, info = step_output
            elif len(step_output) == 5:  # obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
        else:
            # 일반적인 step 결과 처리
            if len(step_result) == 4:  # obs, reward, done, info
                obs, reward, done, info = step_result
            elif len(step_result) == 5:  # obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
        
        # 프레임 저장
        if 'image' in obs:
            img = np.array(obs['image'])
            if img.shape[-1] == 3:
                frames.append(img)
            else:
                frames.append(np.transpose(img, (1, 2, 0)))
        
        # 리워드 누적
        total_reward += reward
        step += 1
        
        # 성공 여부 확인 (Metaworld 환경의 경우)
        if isinstance(info, dict) and 'success' in info and info['success']:
            success = True
            print(f"태스크 성공! 스텝: {step}")
        
        if done:
            break
    
    # 비디오 저장
    print(f"에피소드 종료 - 총 스텝: {step}, 총 리워드: {total_reward}, 성공: {success}")
    
    if len(frames) > 0:
        frames = np.array(frames)
        
        # 비디오 저장을 imageio를 사용해 진행
        try:
            fps = 30
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"비디오가 성공적으로 저장되었습니다: {save_path}")
        except Exception as e:
            print(f"비디오 저장 중 오류 발생: {e}")
    else:
        print("저장할 프레임이 없습니다.")
    
    return {
        "steps": step,
        "total_reward": total_reward,
        "success": success
    }


def main(config):
    # 컴파일 옵션을 비활성화 (테스트에서는 필요하지 않음)
    config.compile = False
    
    # 기본 설정
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    
    # 디렉토리 설정
    logdir = pathlib.Path(config.logdir).expanduser()
    if not logdir.exists():
        raise FileNotFoundError(f"로그 디렉토리를 찾을 수 없습니다: {logdir}")
    
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = logdir / f"test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 비디오 저장 경로
    video_path = test_dir / f"{config.task}_test.mp4"
    
    # 모델 체크포인트 경로
    checkpoint_path = logdir / "latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"모델 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    print(f"테스트 준비 - 모델: {checkpoint_path}")
    
    # LIV 모델 로드
    liv = load_liv()
    liv.eval()
    
    # 타겟 텍스트 임베딩 생성
    text = clip.tokenize(["Agent presses the button."]).to(config.device)
    with torch.no_grad():
        target_text_embedding = liv(input=text, modality="text")
    
    # 환경 생성 - Damy 래퍼 없이 직접 환경 사용
    env = make_env(config, "eval", 0, liv)
    
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # 에이전트 생성 (dreamer.py와 동일한 방식으로)
    from dreamer import Dreamer
    
    dummy_logger = tools.Logger(test_dir, 0)  # 테스트용 더미 로거
    
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        dummy_logger,
        None,  # dataset은 필요 없음
        target_text_embedding,
    ).to(config.device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # strict=False 옵션으로 로드하여 불일치 키 무시
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    
    # 에피소드 실행 및 저장
    results = record_episode(
        agent, 
        env, 
        target_text_embedding, 
        config,
        video_path,
        max_steps=500
    )
    
    # 결과 저장
    with open(test_dir / "results.txt", "w") as f:
        f.write(f"Task: {config.task}\n")
        f.write(f"Steps: {results['steps']}\n")
        f.write(f"Total Reward: {results['total_reward']}\n")
        f.write(f"Success: {results['success']}\n")
    
    print("테스트 완료!")
    print(f"결과: 스텝 = {results['steps']}, 리워드 = {results['total_reward']}, 성공 = {results['success']}")
    print(f"비디오 저장 위치: {video_path}")
    
    # 환경 종료
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--logdir", type=str, required=True, help="학습된 모델이 있는 로그 디렉토리 경로")
    parser.add_argument("--task", type=str, required=True, help="테스트할 태스크 (예: ML1_button-press-v2)")
    args, remaining = parser.parse_known_args()
    
    # configs.yaml 파일에서 설정 로드
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    # 설정 업데이트
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    # 명령행 인자로 설정 덮어쓰기
    config = parser.parse_args(remaining)
    
    # 명시적으로 제공된, 인자를 설정
    if args.logdir:
        config.logdir = args.logdir
    if args.task:
        config.task = args.task
    
    main(config)