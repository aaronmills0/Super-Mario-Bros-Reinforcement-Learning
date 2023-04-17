from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
# env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

import cv2 as cv
from dqn_agent import DqnAgent
from ddqn_agent import DdqnAgent
from dueling_dqn_agent import DuelingDqnAgent
from ppo_agent import PpoAgent
import pandas as pd
import gym
from gym.wrappers import FrameStack
import numpy as np

num_runs = 1

num_episodes = 3000

ACTION_MAP = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left']
]

IMAGE_HEIGHT = 256

IMAGE_WIDTH = 240

HEIGHT = 84

WIDTH = 84

CHANNELS = 3

ACTION_SPACE = len(ACTION_MAP)

"""
Override built-in environment methods such that 4 frames are aggregated into just 1

We want each state to be a set of 4 consecutive frames 

The squashing of the frames is done by the built-in FrameStack wrapper

This custom wrapper skips every 4 frames and returns the last one to avoid overlap

"""
class FrameInterval(gym.Wrapper):

    def __init__(self, env, step_size):
        super().__init__(env)
        self.step_size = step_size

    def step(self, action):
        cumulative_reward = 0.0
        obs = np.zeros((self.step_size, HEIGHT, WIDTH))
        for i in range(self.step_size):
            state, reward, terminated, truncated, info = self.env.step(action)
            obs[i, :, :] = state
            cumulative_reward += reward
            if terminated or truncated:
                break
        return obs, cumulative_reward, terminated, truncated, info

class GrayscaleFrames(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, observation):
        return process_state(observation)

# env = GrayscaleFrames(env)      
# env = FrameInterval(env, step_size=4)
# env = FrameStack(env, num_stack=4)

def process_state(state):
    state = np.stack(state)
    gray = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
    gray = gray[:, :240]
    gray = cv.resize(gray, (HEIGHT, WIDTH))
    return gray

def main():
    # agent = PpoAgent(8192)
    agent = DdqnAgent(8192)
    # agent_type = 'ppo'
    agent_type = 'dqn'

    if agent_type == 'dqn':
        agent.reset()

    load = True

    if (load):
        agent.load("6000")

    action = 0
    reward = 0
    done = False

    STEP_SIZE = 4

    reward_data = [list() for _ in range(num_episodes)]

    win_data = [list() for _ in range(num_episodes)]

    score_data = [list() for _ in range(num_episodes)]

    time_data = [list() for _ in range(num_episodes)]

    for i in range(num_runs):

        for j in range(num_episodes):

            print(f"Run: {i} Episode: {j}")

            state = env.reset()

            state = state[0]

            state = process_state(state)

            obs = np.zeros((STEP_SIZE, HEIGHT, WIDTH), dtype=np.float32)
            obs[0, :, :] = state
            action = np.random.randint(ACTION_SPACE)
            cumulative_reward = 0.0
            for k in range(1, STEP_SIZE):
                state, reward, terminated, truncated, info = env.step(action)
                state = process_state(state)
                obs[k, :, :] = state
                cumulative_reward += reward
                if terminated or truncated:
                    break

            while True:
                
                if agent_type == 'ppo':
                    action = agent.get_action(obs, reward, done)
                else: #dqn
                    action = agent.get_action(obs, action, reward, done)

                obs = np.zeros((STEP_SIZE, HEIGHT, WIDTH), dtype=np.float32)
                cumulative_reward = 0.0
                for k in range(STEP_SIZE):
                    state, reward, terminated, truncated, info = env.step(action)
                    state = process_state(state)
                    obs[k, :, :] = state
                    cumulative_reward += reward
                    if terminated or truncated:
                        break
                
                done = terminated or truncated

                reward_data[j].append(cumulative_reward)

                win_data[j].append(info["flag_get"])

                score_data[j].append(info["score"])

                time_data[j].append(info["time"])

                # print(f"action: {action} reward: {reward}")
                if done:
                    if agent_type == 'dqn':
                        agent.reset()
                    break
                
    env.close()

    pd.DataFrame(reward_data).to_csv("./data/ddqn_rewards_9000.csv")
    pd.DataFrame(win_data).to_csv("./data/ddqn_wins_9000.csv")
    pd.DataFrame(score_data).to_csv("./data/ddqn_score_9000.csv")
    pd.DataFrame(time_data).to_csv("./data/ddqn_time_9000.csv")

    agent.save("9000")


if __name__ == "__main__":
    main()