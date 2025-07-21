import gymnasium as gym

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
    
