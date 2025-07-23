'''
Quantum Fourier Transform Benchmark Program - Environment Files
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import random
import numpy as np

class Environment:
    """
    Class to encapsulate the RL environment (FrozenLake).
    """
    env = None

    def make_env(self, render_mode=None, is_slippery=False, map_name="4x4"):
        """
        Create and initialize a FrozenLake-v1 environment.

        Args:
            render_mode (str or None): Set to "human" for visual rendering. (requires additional package `pip install "gymnasium[toy-text]"`)
            is_slippery (bool): Whether the ice is slippery (stochastic transitions).
            map_name (str): Either "4x4" or "8x8".

        Returns:
            None. Sets self.env to the created environment.
        """
        self.env = gym.make(
            "FrozenLake-v1", 
            render_mode=render_mode, 
            is_slippery=is_slippery, 
            map_name=map_name
        )
    
    def set_max_steps_per_episode(self, max_episode_steps=20):
        """
        Limit the maximum number of steps per episode using TimeLimit wrapper.

        Args:
            max_episode_steps (int): Maximum steps allowed per episode.

        Returns:
            None. Updates self.env to be wrapped with TimeLimit.
        """
        self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)
    
    def reset(self, seed=0):
        """
        Reset the environment to the initial state.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            obs (int): The initial observation/state after reset.
        """
        obs, _ = self.env.reset()
        print(f"Environment reset: obs: {obs}")
        return obs

    def sample(self):
        """
        Sample a random action from the action space.

        Returns:
            action (int): A randomly sampled action.
        """
        return self.env.action_space.sample()
    
    def step(self, action):
        """
        Take a step in the environment using the given action.

        Args:
            action (int): The action to take.

        Returns:
            tuple: (next_obs, reward, terminated, truncated, info)
            next_obs (int): Next observation/state.
            reward (float): Reward received after taking the action.
            terminated (bool): Whether the episode has ended (success/failure).
            truncated (bool): Whether the episode was truncated (e.g., time limit).
            info (dict): Additional information.
        """
        return self.env.step(action)
    
    def get_num_of_actions(self):
        """
        Get the number of possible actions in the environment.

        Returns:
            n_actions (int): Number of actions.
        """
        return self.env.action_space.n
    
    def get_observation_size(self):
        """
        Get the size of the observation space (number of states).

        Returns:
            n_states (int): Number of possible states.
        """
        return self.env.observation_space.n
    
class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling experience tuples.
    """
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of items in the buffer.

        Returns:
            None.
        """
        self.capacity = capacity  # Maximum number of items in the buffer
        self.buffer = []          # List to store experience tuples
        self.pointer = 0          # Pointer for circular buffer replacement
    
    def add_buffer_item(self, obs, next_obs, action, reward, done):
        """
        Add a new experience tuple to the buffer.

        Args:
            obs: Current observation/state.
            next_obs: Next observation/state after action.
            action: Action taken.
            reward: Reward received.
            done: Boolean indicating if the episode ended.

        Returns:
            None.
        """
        if len(self.buffer) < self.capacity:
            # If buffer not full, append new item
            self.buffer.append((obs, next_obs, action, reward, done))
        else:
            # If buffer full, overwrite the oldest item (circular buffer)
            self.buffer[self.pointer] = (obs, next_obs, action, reward, done)
            self.pointer += 1
            self.pointer = self.pointer % self.capacity # Always keeps the pointer in range [0, self.capacity)
    
    def sample_batch_from_buffer(self, batch_size: int):
        """
        Randomly sample a batch of experience tuples from the buffer.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            tuple: (obs, next_obs, actions, rewards, dones)
                obs (np.ndarray): Batch of observations.
                next_obs (np.ndarray): Batch of next observations.
                actions (np.ndarray): Batch of actions.
                rewards (np.ndarray): Batch of rewards.
                dones (np.ndarray): Batch of done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)

        return (
            np.array(obs),
            np.array(next_obs),
            np.array(actions),
            np.array(rewards),
            np.array(dones)
        )
