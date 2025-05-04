import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import metaworld
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy

from metaworld_wrapper import MetaWorldEnvWrapper
from liv import load_liv
import clip
import imageio
from PIL import Image

def simulate_episode(seed=42, save_dir="results_practice2"):
    """
    Metaworld의 button-press 태스크에서 하나의 에피소드를 시뮬레이션하고 
    LIV 임베딩과 텍스트 타겟 간의 코사인 유사도를 계산하여 시각화합니다.
    
    Args:
        seed: 환경의 시드
        save_dir: 결과를 저장할 디렉토리
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # LIV 모델 로드
    liv = load_liv()
    liv.eval()
    
    # 텍스트 임베딩 생성
    text_prompt = "Robot pushes button"
    text_tokens = clip.tokenize([text_prompt]).to('cuda:0')
    with torch.no_grad():
        text_embedding = liv(input=text_tokens, modality="text")
    
    # MetaWorld 환경 래퍼 생성 - LIV 모델 전달
    env = MetaWorldEnvWrapper(task_name='button-press-v2', liv=liv, seed=seed, mode="train")
    # 초기 설정 변경 (버튼 위치와 손 위치)
    
    custom_config = {
        'obj_init_pos': [0.0, 0.0, 0.0],  # x, y, z 좌표
        'hand_init_pos': [0.05, 0.6, 0.115]     # x, y, z 좌표
    }
    if custom_config:
        print("사용자 정의 초기 설정 적용:", custom_config)
        if 'obj_init_pos' in custom_config:
            env.env.obj_init_pos = np.array(custom_config['obj_init_pos'], dtype=np.float32)
        if 'hand_init_pos' in custom_config:
            env.env.hand_init_pos = np.array(custom_config['hand_init_pos'], dtype=np.float32)
    
    # 정책 로드
    policy = SawyerButtonPressV2Policy()
    #policy = SawyerDoorOpenV2Policy()

    # 비디오 프레임 및 유사도 저장을 위한 리스트
    frames = []
    similarities = []
    steps = []
    
    # 환경 초기화
    obs = env.reset()
    done = False
    step_count = 0
    
    print("에피소드 시뮬레이션 시작...")
    
    # 에피소드 실행
    while not done and step_count < env.env.max_path_length:
        # 화면 렌더링 및 저장
        frame = env.render()
        frames.append(frame)
        print(env.env.model.body("hand").pos)
        # 정책에서 액션 얻기 (MetaWorld 원본 observation 필요)
        action = policy.get_action(env.env._get_obs())
        
        # 환경에서 한 스텝 진행
        next_obs, reward, done, info = env.step(action)
        
        # 임베딩 가져오기 (wrapper에서 이미 계산됨)
        obs_embedding = torch.from_numpy(next_obs["target_image_embedding"]).to('cuda:0')
        
        # 코사인 유사도 계산
        text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        obs_embedding_norm = obs_embedding / obs_embedding.norm(dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(
            obs_embedding_norm, text_embedding_norm, dim=-1
        ).item()
        
        # 유사도 저장
        similarities.append(similarity)
        steps.append(step_count)
        
        # 정보 출력
        print(f"Step {step_count}: Similarity = {similarity:.4f}, Success = {info['success'] if 'success' in info else 'N/A'}")
        
        # 다음 스텝 준비
        obs = next_obs
        step_count += 1
        
        # 성공 여부 확인
        if 'success' in info and info['success'] == 1:
            print(f"성공! 스텝 {step_count}에서 버튼 누르기 완료.")
            # 성공한 후에도 몇 프레임 더 기록
            for _ in range(10):
                if step_count < env.env.max_path_length:
                    frame = env.render()
                    frames.append(frame)
                    step_count += 1
            break
    
    print(f"에피소드 종료. 총 {step_count} 스텝 실행.")
    
    # 비디오 저장
    video_path = os.path.join(save_dir, 'button_press_episode.mp4')
    imageio.mimsave(video_path, frames, fps=15)
    print(f"비디오 저장 완료: {video_path}")
    
    # 유사도 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(steps, similarities, marker='o', linestyle='-', markersize=4)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # 0 기준선
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity between LIV Embedding and Text Target\nText: "{text_prompt}"')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 최대/최소 유사도 표시
    if similarities:
        max_sim = max(similarities)
        max_idx = similarities.index(max_sim)
        plt.annotate(f'Max: {max_sim:.4f}', 
                    xy=(steps[max_idx], max_sim), 
                    xytext=(steps[max_idx]+5, max_sim), 
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    sim_path = os.path.join(save_dir, 'similarity_plot.png')
    plt.savefig(sim_path)
    print(f"유사도 그래프 저장 완료: {sim_path}")
    
    # 텍스트로 유사도 저장
    data_path = os.path.join(save_dir, 'similarity_data.csv')
    with open(data_path, 'w') as f:
        f.write('Step,Similarity\n')
        for step, sim in zip(steps, similarities):
            f.write(f'{step},{sim}\n')
    print(f"유사도 데이터 저장 완료: {data_path}")
    
    # 환경 정리
    env.close()
    
    return {
        'video_path': video_path,
        'similarity_path': sim_path,
        'data_path': data_path,
        'steps': steps,
        'similarities': similarities
    }

if __name__ == "__main__":
    results = simulate_episode()
    
    # 추가 분석 메트릭
    if results['similarities']:
        avg_sim = sum(results['similarities']) / len(results['similarities'])
        max_sim = max(results['similarities'])
        min_sim = min(results['similarities'])
        final_sim = results['similarities'][-1]
        
        print("\n===== 유사도 분석 =====")
        print(f"평균 유사도: {avg_sim:.4f}")
        print(f"최대 유사도: {max_sim:.4f}")
        print(f"최소 유사도: {min_sim:.4f}")
        print(f"최종 유사도: {final_sim:.4f}")