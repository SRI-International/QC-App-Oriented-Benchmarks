import gymnasium as gym

def make_env(render_mode=None, is_slippery=True, map_name="4x4"):
    """
    Creates and returns a FrozenLake-v1 environment.

    Args:
        render_mode (str or None): Set to "human" for visual rendering. (requires additional package `pip install "gymnasium[toy-text]"`)
        is_slippery (bool): Whether the ice is slippery (stochastic transitions).
        map_name (str): Either "4x4" or "8x8".

    Returns:
        env (gym.Env): Initialized FrozenLake-v1 environment.
    """
    env = gym.make(
        "FrozenLake-v1", 
        render_mode=render_mode, 
        is_slippery=is_slippery, 
        map_name=map_name
    )
    return env

