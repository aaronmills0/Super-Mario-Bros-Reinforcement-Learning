from dqn import DQN
from replay_buffer import ReplayBuffer
import replay_buffer
import numpy as np
import torch.optim as optim
import torch.nn as nn
import copy
import torch
import random

ACTION_SPACE = 7

HEIGHT =  84

WIDTH = 84

CHANNELS = 4

LR = 0.0001

class DdqnAgent:

    def __init__(self, buffer_capacity, epsilon=0.3):

        self.q = DQN(ACTION_SPACE)

        self.target = DQN(ACTION_SPACE)

        self.replaybuffer = ReplayBuffer(buffer_capacity)

        self.epsilon = 0.3

        self.gamma = 0.99

        self.batch_size = 128

        self.train_delay = 256

        self.update_delay = 1024

        self.train_counter = 0

        self.update_counter = 0

        self.optimizer = optim.Adam(params=self.q.parameters(), lr=LR)

        self.prev_state = None

        self.step_counter = 0
 
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_gamma(self, gamma):
        self.gamma = gamma

    def epsilon_greedy(self, output):

        num = random.random() # random number between 0 and 1

        if (num < self.epsilon):
            return np.random.randint(ACTION_SPACE)

        return torch.argmax(output).item()
    
    def train(self):

        self.q.train()

        self.target.eval()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replaybuffer.sample(self.batch_size)

        state_tensor = state_batch
        state_tensor = state_tensor.reshape(self.batch_size, CHANNELS, HEIGHT, WIDTH)

        next_state_tensor = next_state_batch
        next_state_tensor = next_state_tensor.reshape(self.batch_size, CHANNELS, HEIGHT, WIDTH)

        action_tensor = action_batch
        action_tensor = action_tensor.reshape(1, self.batch_size)

        reward_tensor = reward_batch
        reward_tensor = reward_tensor.reshape(self.batch_size)

        done_tensor = done_batch.apply_(lambda x: 1 - x)
        done_tensor = done_tensor.reshape(self.batch_size)

        q_output_values = self.q(state_tensor).gather(1, action_tensor)

        target_output_values = torch.zeros(self.batch_size)

        with torch.no_grad():
            q_ns_max_output_actions = self.q(next_state_tensor).detach().max(1)[1]
            q_ns_max_output_actions = q_ns_max_output_actions.reshape(1, self.batch_size)

        with torch.no_grad():
            target_output_values = (done_tensor * (self.target(next_state_tensor).detach().gather(1, q_ns_max_output_actions)))*self.gamma + reward_tensor

        criterion = nn.MSELoss()

        loss = criterion(q_output_values, target_output_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for params in self.q.parameters():
            params.grad.data.clamp_(-1, 1)
        self.optimizer.step()      

    def update(self):
        self.target.load_state_dict(self.q.state_dict())

    def reset(self):
        self.prev_state = None

    def save(self, msg=""):
        q_state = {
                'state_dict': self.q.state_dict(),
                'optimizer': self.optimizer.state_dict(),
        }
        torch.save(q_state, "./models/ddqn_q_model" + msg + ".pt")
        target_state = {
                'state_dict': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(),
        }
        torch.save(target_state, "./models/ddqn_target_model" + msg + ".pt")

    def load(self, msg=""):
        checkpoint_q = torch.load("./models/ddqn_q_model" + msg + ".pt")
        self.q.load_state_dict(checkpoint_q['state_dict'])
        self.optimizer.load_state_dict(checkpoint_q['optimizer'])
        self.q.train()
        checkpoint_target = torch.load("./models/ddqn_target_model" + msg + ".pt")
        self.target.load_state_dict(checkpoint_target['state_dict'])
        self.optimizer.load_state_dict(checkpoint_target['optimizer'])
        self.target.train()

    def get_action(self, state, action, reward, done):
        self.step_counter += 1

        if (self.step_counter % 1000000 == 0):
            self.save(msg="_" + str(self.step_counter))

        if (self.replaybuffer.size == 0):  
            self.prev_state = state

        if self.prev_state is None:
            self.prev_state = state
        
        self.replaybuffer.add_entry(self.prev_state, action, reward, state, done)

        self.prev_state = state
    
    
        state = torch.from_numpy(state.astype(np.float32))

        state = state.reshape(1, 4, 84, 84)

        output = self.q.forward(state)

        action = self.epsilon_greedy(output)

        if self.train_counter >= self.train_delay:
            self.train_counter = -1
            self.train()
        self.train_counter += 1

        if self.update_counter >= self.update_delay:
            self.update_counter = -1
            self.update()
        self.update_counter += 1

        return action















