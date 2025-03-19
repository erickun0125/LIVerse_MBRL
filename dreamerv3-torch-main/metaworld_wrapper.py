import gym
import numpy as np
import metaworld
from gym import spaces
import random
from liv import load_liv
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
    A Gym-style wrapper for a MetaWorld task (using the MT1 benchmark).
    This wrapper makes the environment compatible with DreamerV3 by
    returning observations as a dictionary with keys:
      - "image": the raw state observation (note: not an actual image)
      - "discount": a discount factor (1.0 if not terminal, 0.0 if done)
      - "is_first": a flag that is 1 at the beginning of an episode and 0 thereafter
      - "is_terminal": a flag that is 1 if the current state is terminal, else 0

    Usage:
      env = MetaWorldEnvWrapper(task_name='reach-v1', task_index=0, seed=42)
      obs = env.reset()
      next_obs, reward, done, info = env.step(action)
    """
    def __init__(self, task_name, seed=None,mode="train"):
        super(MetaWorldEnvWrapper, self).__init__()
        self.task_name = task_name
        self.liv = load_liv()
        self.liv.eval()
        self.transform = T.Compose([T.ToTensor()])
        if seed is not None:
            random.seed(seed)
        # Create the MT1 benchmark instance (a single-task benchmark)
        ml1 = metaworld.ML1(task_name)
        
        if task_name not in ml1.train_classes:
            raise ValueError(f"Task '{task_name}' not found in ML1 benchmark.")
        # Instantiate the environment for the given task.
         
        if "train" in mode:
            self.env = ml1.train_classes[task_name]()
            task = random.choice(ml1.train_tasks)
            self.env.set_task(task)
        elif "test" in mode:
            self.env = ml1.test_classes[task_name]()
            task = random.choice(ml1.test_tasks)
            self.env.set_task(task)
        elif "eval" in mode:
            #temporary for now
            self.env = ml1.test_classes[task_name]()
            task = random.choice(ml1.test_tasks)
            self.env.set_task(task)

        if seed is not None:
            self.env.seed(seed)
        """
        # Wrap the underlying observation space into a Dict space with the required keys.
        # Here we assume the underlying env uses a Box for state observations.
        obs_space = getattr(self.env, 'observation_space', None)
        #print(obs_space)
        if obs_space is None:
            # If not available, sample one observation to infer the shape.
            sample_obs = self.env.reset()
            obs_shape = np.array(sample_obs).shape
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        else:
            # Convert gymnasium space to gym space if necessary.
            obs_space = convert_to_gym_space(obs_space)
        self.observation_space = spaces.Dict({
            "image": obs_space,
            "discount": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
        })
        
        """
        # Get a sample rendered image from the environment.
        sample_image = self.env.render()
        # Define the image space based on the sample.
        # We assume pixel values are in [0, 255] and of type uint8.
        image_space = spaces.Box(low=0, high=255, shape=sample_image.shape, dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "image": image_space,
            "discount": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
        })
        self.action_space = self.env.action_space
        # Internal flag to mark the first timestep of an episode.
        self.first = True

    def reset(self):
        
        self.env.reset()
        raw_obs = self.render()
        self.first = True
        return self._process_obs(raw_obs, is_done=False)

    def step(self, action):
        raw_obs, reward,_, done, info = self.env.step(action)
        raw_obs = self.render()
        #reward dummy value of 0
        return self._process_obs(raw_obs, is_done=done), 0, done, info

    def _process_obs(self, obs, is_done):
        """
        Converts the raw observation into a dictionary with the keys expected by DreamerV3.
        """
        pil_image = self.transform(Image.fromarray(obs)).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            target_image_embedding= self.liv(input=pil_image, modality="vision")
        result = {
            # We call the key "image" even though this is a state vector.
            "image": np.array(obs),
            # Provide a discount factor: 1.0 if not done, 0.0 if terminal.
            "discount": np.array([0.0 if is_done else 1.0], dtype=np.float32),
            # 'is_first' is 1 only on the first step after a reset.
            "is_first": np.array([1] if self.first else [0], dtype=np.int8),
            # 'is_terminal' indicates whether the current step ended the episode.
            "is_terminal": np.array([1] if is_done else [0], dtype=np.int8),
        "target_image_embedding":target_image_embedding.cpu().numpy(),}
        self.first = False
        return result

    def render(self, mode="rgb_array"):
        return self.env.render()

    def close(self):
        self.env.close()
