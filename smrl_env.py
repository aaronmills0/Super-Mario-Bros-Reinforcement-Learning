from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
import cv2 as cv
from dqn_agent import DqnAgent

num_runs = 10

num_episodes = 5000

ACTION_MAP = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

IMAGE_HEIGHT = 256

IMAGE_WIDTH = 240

HEIGHT = 84

WIDTH = 84

CHANNELS = 3

ACTION_SPACE = len(ACTION_MAP)

def process_state(state):
    gray = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
    gray = gray[:, :240]
    gray = cv.resize(gray, (HEIGHT, WIDTH))
    return gray

def main():
    agent = DqnAgent(16384)

    action = 0
    reward = 0
    done = False

    counter = 0

    for i in range(num_runs):


        for j in range(num_episodes):

            state = env.reset()

            while True:

                frame = process_state(state)

                action = agent.get_action(frame, action, reward, done)

                state, reward, done, info = env.step(action)

                # print(f"action: {action} reward: {reward}")

                if done:
                    break

                env.render()
    env.close()

if __name__ == "__main__":
    main()