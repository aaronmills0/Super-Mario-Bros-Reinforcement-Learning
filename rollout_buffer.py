from collections import deque
import itertools
import numpy as np
import torch

HEIGHT = 84
WIDTH = 84
CHANNELS = 4

class RolloutBuffer:

    def __init__(self, capacity):
        self.state_buffer = deque()

        self.action_buffer = deque()

        self.reward_buffer = deque()

        self.done_buffer = deque()

        self.value_buffer = deque()

        self.log_probability_buffer = deque()

        self.returns_buffer = deque()

        self.advantage_buffer = deque()

        self.size = 0

        self.trajectory_pointer = 0

        self.capacity = capacity

        self.latest = 0

    def add_entry(self, state, action, reward, value, log_probability, done):
        self.state_buffer.appendleft(state)
        self.action_buffer.appendleft(action)
        self.reward_buffer.appendleft(reward)
        self.done_buffer.appendleft(done)
        self.value_buffer.appendleft(value)
        self.log_probability_buffer.appendleft(log_probability)
        self.trajectory_pointer += 1
        self.size += 1
        if (self.size > self.capacity):
            self.remove_trajectory()
        if self.trajectory_pointer > self.size:
            self.trajectory_pointer = self.size

    # Remove the oldest trajectory
    def remove_trajectory(self):
        done = False
        while not done and self.size > 0:
            self.state_buffer.pop()
            self.action_buffer.pop()
            self.reward_buffer.pop()
            done = self.done_buffer.pop()
            self.value_buffer.pop()
            self.log_probability_buffer.pop()
            self.advantage_buffer.pop()
            self.returns_buffer.pop()
            self.size -= 1

    def complete_trajectory(self, gamma):

        discounted_reward = 0
        returns = []
        values = []
        
        for reward, value in zip(itertools.islice(self.reward_buffer, 0, self.trajectory_pointer), itertools.islice(self.value_buffer, 0, self.trajectory_pointer)):
            discounted_reward += reward
            returns.append(discounted_reward)
            discounted_reward *= gamma

            values.append(value)
        
        for r, v in zip(reversed(returns), reversed(values)):
            self.returns_buffer.appendleft(r)
            self.advantage_buffer.appendleft(r - v)
        
        self.trajectory_pointer = 0;
        self.latest = self.size

    def sample(self, batch_size):
        indexes = np.arange(len(self.returns_buffer))



        samples = np.random.choice(indexes, batch_size)

        states = []
        actions = []
        values = []
        returns = []
        advantages = []
        log_probabilities = []

        for index in samples:
            states.append(self.state_buffer[index])
            actions.append(self.action_buffer[index])
            values.append(self.value_buffer[index])
            returns.append(self.returns_buffer[index])
            advantages.append(self.advantage_buffer[index])
            log_probabilities.append(self.log_probability_buffer[index])
        
        return torch.tensor(states), torch.tensor(actions), torch.tensor(values), torch.tensor(returns), torch.tensor(advantages), torch.tensor(log_probabilities)

