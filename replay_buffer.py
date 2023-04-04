from collections import deque
import numpy as np

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

        self.context = deque()

        self.size = 0

        self.num_frames = 0

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
    
    def add_context(self, frame):
        if (self.size >= self.capacity):
            self.context.pop()
            self.num_frames -= 1
        else:
            self.num_frames += 1
        
        self.context.appendleft(frame)

    def get_state(self):
        state = np.zeros((CHANNELS, HEIGHT, WIDTH))
        if len(self.context) < CHANNELS:
            raise ValueError(f"Not enough frames in the context buffer: {len(self.context)} frames out of {self.capacity}")
        state[0, :, :] = self.context[0]
        state[1, :, :] = self.context[1]
        state[2, :, :] = self.context[2]
        state[3, :, :] = self.context[3]
        return state

    def sample(self, batch_size):
        indices = np.random.choice(len(self.state_buffer), batch_size)
        state_arr = []
        action_arr = []
        reward_arr = []
        next_state_arr = []
        done_arr = []
        for i in indices:
            state_arr.append(self.state_buffer[i])
            action_arr.append(self.action_buffer[i])
            reward_arr.append(self.reward_buffer[i])
            next_state_arr.append(self.next_state_buffer[i])
            done_arr.append(self.done_buffer[i])
        return np.array(state_arr), np.array(action_arr), np.array(reward_arr), np.array(next_state_arr), np.array(done_arr)