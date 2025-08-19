import gym
import numpy as np
import metaworld
from gym import spaces
import random
import torch
import torchvision.transforms as T
from PIL import Image

def convert_to_gym_space(space):
    """
    Converts a gymnasium space to a gym space.
    """
    # If it's already a gym space, return it.
    if isinstance(space, gym.Space):
        return space
    try:
        import gymnasium as gymn
        # If the space is a Box from gymnasium, convert it.
        if isinstance(space, gymn.spaces.Box):
            return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        # If it's a Dict space, recursively convert its subspaces.
        if isinstance(space, gymn.spaces.Dict):
            converted = {key: convert_to_gym_space(subspace) for key, subspace in space.spaces.items()}
            return spaces.Dict(converted)
        # If it's a Discrete space from gymnasium, convert it.
        if isinstance(space, gymn.spaces.Discrete):
            return spaces.Discrete(n=space.n)
        # Add more conversions as needed...
    except ImportError:
        pass
    raise ValueError("Unable to convert space: " + str(space))


class MetaWorldEnvWrapper(gym.Env):
    """
    A Gym-style wrapper for a MetaWorld task (using the ML1 benchmark).
    This wrapper makes the environment compatible with DreamerV3 by
    returning observations as a dictionary with keys:
      - "image": 렌더링된 이미지 데이터
      - "raw_obs": 환경에서 반환된 원시 상태 데이터
      - "discount": 할인 인자 (1.0 if not terminal, 0.0 if done)
      - "is_first": 에피소드 시작 여부 플래그 (1 at the beginning, 0 thereafter)
      - "is_terminal": 종료 상태 여부 플래그 (1 if terminal, 0 otherwise)
      - "target_image_embedding": LIV 모델로 생성한 이미지 임베딩

    Usage:
      env = MetaWorldEnvWrapper(task_name='reach-v1', liv=liv_model, seed=42)
      obs = env.reset()
      next_obs, reward, done, info = env.step(action)
    """
    def __init__(self, task_name, liv, seed=None, mode="train"):
        super(MetaWorldEnvWrapper, self).__init__()
        self.task_name = task_name
        self.liv = liv
        self.liv.eval()
        self.transform = T.Compose([T.ToTensor()])
        if seed is not None:
            random.seed(seed)
            
        # Create the ML1 benchmark instance (a single-task benchmark)
        ml1 = metaworld.ML1(task_name)
        
        if task_name not in ml1.train_classes:
            raise ValueError(f"Task '{task_name}' not found in ML1 benchmark.")
            
        # Instantiate the environment for the given task.
        if "train" in mode:
            self.env = ml1.train_classes[task_name](camera_id=1)
            task = random.choice(ml1.train_tasks)
            self.env.set_task(task)
        elif "test" in mode:
            self.env = ml1.train_classes[task_name](camera_id=1)
            task = random.choice(ml1.test_tasks)
            self.env.set_task(task)
        elif "eval" in mode:
            # temporary for now
            self.env = ml1.train_classes[task_name](camera_id=1)
            task = random.choice(ml1.test_tasks)
            self.env.set_task(task)

        if seed is not None:
            self.env.seed(seed)
            
        # Get a sample rendered image and raw observation
        sample_raw_obs, _ = self.env.reset()
        sample_image = self.env.render()
        
        # Define observation spaces
        obs_shape = np.array(sample_raw_obs).shape
        #print("wow : ", obs_shape)
        raw_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        image_space = spaces.Box(low=0, high=255, shape=sample_image.shape, dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "image": image_space,
            "raw_obs": raw_obs_space,
            "discount": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
        })
        
        self.action_space = self.env.action_space
        # Internal flag to mark the first timestep of an episode.
        self.first = True

    def reset(self):
        raw_obs, _ = self.env.reset()
        img_obs = self.render()
        #print("raw_obs : ", raw_obs)
        self.first = True
        return self._process_obs(raw_obs, img_obs, is_done=False)

    def step(self, action):
        raw_obs, reward, _, done, info = self.env.step(action)
        img_obs = self.render()
        # reward is dummy value of 0 as it will be computed by LIV
        return self._process_obs(raw_obs, img_obs, is_done=done), 0, done, info

    def _process_obs(self, raw_obs, img_obs, is_done):
        """
        Converts the observations into a dictionary with the keys expected by DreamerV3.
        
        Args:
            raw_obs: 환경의 step()에서 반환한 원시 상태 데이터
            img_obs: 환경의 render()에서 반환한 이미지 데이터
            is_done: 에피소드 종료 여부
            
        Returns:
            observation 딕셔너리
        """
        pil_image = self.transform(Image.fromarray(img_obs)).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            target_image_embedding = self.liv(input=pil_image, modality="vision")
            
        result = {
            # 시각적 관측값 (render()의 결과)
            "image": np.array(img_obs),
            # 원시 상태 데이터 (step()의 반환값)
            "raw_obs": np.array(raw_obs, dtype=np.float32),
            # 할인 인자: 1.0 (not done), 0.0 (terminal)
            "discount": np.array([0.0 if is_done else 1.0], dtype=np.float32),
            # 'is_first'는 에피소드 첫 스텝에서만 1
            "is_first": np.array([1] if self.first else [0], dtype=np.int8),
            # 'is_terminal'은 현재 스텝이 종료 상태일 때 1
            "is_terminal": np.array([1] if is_done else [0], dtype=np.int8),
            # LIV 모델로 생성한 이미지 임베딩
            "target_image_embedding": target_image_embedding.cpu().numpy(),
        }
        self.first = False
        return result

    def render(self, mode="rgb_array"):
        orig_img = self.env.render()
        if hasattr(self.env, 'camera_id') and self.env.camera_id not in [None, 0]:
            # 이미지 상하 반전
            orig_img = np.flipud(orig_img)
        return orig_img
    
    def close(self):
        self.env.close()