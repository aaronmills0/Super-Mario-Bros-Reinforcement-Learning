from collections import deque
import numpy as np
import torch

HEIGHT = 84
WIDTH = 84
CHANNELS = 4

class ReplayBuffer:

    def __init__(self, capacity):

        self.state_buffer = deque()

        self.action_buffer = deque()

        self.reward_buffer = deque()

        self.next_state_buffer = deque()

        self.done_buffer = deque()

        self.size = 0

        self.capacity = capacity

    
    def add_entry(self, state, action, reward, next_state, done):
        if (self.size >= self.capacity):
            self.state_buffer.pop()
            self.action_buffer.pop()
            self.reward_buffer.pop()
            self.next_state_buffer.pop()
            self.done_buffer.pop()
            self.size -= 1
        else:
            self.size += 1
        self.state_buffer.appendleft(state)
        self.action_buffer.appendleft(action)
        self.reward_buffer.appendleft(reward)
        self.next_state_buffer.appendleft(next_state)
        self.done_buffer.appendleft(done)

    def get_state(self, i=0):
        state = np.zeros((CHANNELS, HEIGHT, WIDTH))
        if len(self.next_state_buffer) < CHANNELS:
            raise ValueError(f"Not enough frames in the context buffer: {len(self.context)} frames out of {self.capacity}")
        state[0, :, :] = self.next_state_buffer[0]
        state[1, :, :] = self.next_state_buffer[1]
        state[2, :, :] = self.next_state_buffer[2]
        state[3, :, :] = self.next_state_buffer[3]
        return state

    # Changes were made here
    def sample(self, batch_size):
        indexes = np.arange(self.size)
        samples = np.random.choice(indexes, batch_size)
        state_arr = []
        action_arr = []
        reward_arr = []
        next_state_arr = []
        done_arr = []
        for i in samples:
            state_arr.append(self.state_buffer[i])
            action_arr.append(self.action_buffer[i])
            reward_arr.append(self.reward_buffer[i])
            next_state_arr.append(self.next_state_buffer[i])
            done_arr.append(self.done_buffer[i])
        return torch.from_numpy(np.array(state_arr, dtype=np.float32)), torch.from_numpy(np.array(action_arr)), torch.from_numpy(np.array(reward_arr, dtype=np.float32)), torch.from_numpy(np.array(next_state_arr, dtype=np.float32)), torch.from_numpy(np.array(done_arr))