import gymnasium as gym
import numpy as np


def create_env():
    env = gym.make("FrozenLake-v1")
    return env

def choose_action(q_values, epsilon, env):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_values)
    
def isDone(state, env, steps, max_steps):
    if state == env.goal_state:
        return True
    elif state == env.hole_state:
        return True
    elif steps > max_steps:
        return True
    else:
        return False
    


def train_q_values(env, q_values, num_episodes, alpha, gamma, epsilon, max_steps):
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        while not done:
            action = choose_action(q_values, epsilon, env)
            next_state, reward, done, info = env.step(action)
            q_values[state, action] = (1 - alpha) * q_values[state, action] + alpha * (reward + gamma * np.max(q_values[next_state]))
