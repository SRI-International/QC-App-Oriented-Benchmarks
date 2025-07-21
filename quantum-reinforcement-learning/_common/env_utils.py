import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import random
import numpy as np

class Environment:
    env = None

    def make_env(self, render_mode=None, is_slippery=False, map_name="4x4"):
        """
        Creates and returns a FrozenLake-v1 environment.

        Args:
            render_mode (str or None): Set to "human" for visual rendering. (requires additional package `pip install "gymnasium[toy-text]"`)
            is_slippery (bool): Whether the ice is slippery (stochastic transitions).
            map_name (str): Either "4x4" or "8x8".

        Returns:
            env (gym.Env): Initialized FrozenLake-v1 environment.
        """
        self.env = gym.make(
            "FrozenLake-v1", 
            render_mode=render_mode, 
            is_slippery=is_slippery, 
            map_name=map_name
        )
    
    def set_max_steps_per_episode(self, max_episode_steps = 20):
        self.env = TimeLimit(self.env, max_episode_steps = max_episode_steps)
    

    def reset(self, seed = 0):
        obs, _ = self.env.reset()
        print(f"Environment reset: obs: {obs}")
        return obs

    def sample(self):
        return self.env.action_space.sample()
    
    def step(self, action):
        return self.env.step(action)
    
    def get_num_of_actions(self):
        return self.env.action_space.n
    
    def get_observation_size(self):
        return self.env.observation_space.n
    
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0
    
    def add_buffer_item(self, obs, next_obs, action, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, next_obs, action, reward, done))
        else:
            self.buffer[self.pointer] = (obs, next_obs, action, reward, done)
            self.pointer += 1
            self.pointer = self.pointer % self.capacity # Always keeps the pointer in range [0, self.capacity)
    
    def sample_batch_from_buffer(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)

        return (
            np.array(obs),
            np.array(next_obs),
            np.array(actions),
            np.array(rewards),
            np.array(dones)
        )
