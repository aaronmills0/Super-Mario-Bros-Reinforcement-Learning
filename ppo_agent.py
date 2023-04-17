import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import random
from ppo_actor import Actor
from ppo_critic import Critic
from rollout_buffer import RolloutBuffer
import rollout_buffer
from math import log
from torch.distributions import Categorical

ACTION_SPACE = 7

HEIGHT =  84

WIDTH = 84

CHANNELS = 4

LR = 0.00025

class PpoAgent:

    def __init__(self, buffer_capacity):

        self.actor = Actor(ACTION_SPACE)

        self.critic = Critic()

        self.rolloutbuffer = RolloutBuffer(buffer_capacity)

        self.gamma = 0.99

        self.batch_size = 256

        self.train_delay = 256

        self.train_counter = 0

        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=LR)

        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=LR)

        self.clip = 0.2

        self.step_counter = 0

        self.trajectory_complete = False

    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_clip(self, clip):
        self.clip = clip
        
    def log_probabilities(self, logits):
        log_p = []
        for p in logits:
            log_p.append(log(p))
        return log_p
    
    def train(self):
        self.actor.train()

        self.critic.train()

        state_batch, action_batch, value_batch, return_batch, advantage_batch, log_probability_batch = self.rolloutbuffer.sample(self.batch_size)

        state_tensor = state_batch
        state_tensor = state_tensor.reshape(self.batch_size, CHANNELS, HEIGHT, WIDTH)

        action_tensor = action_batch
        action_tensor = action_tensor.reshape(1, self.batch_size)

        value_tensor = value_batch
        value_tensor = value_tensor.reshape(self.batch_size)

        return_tensor = return_batch
        return_tensor = return_tensor.reshape(self.batch_size, 1)
        # Normalize the returns
        returns = (return_tensor - return_tensor.mean()) / (return_tensor.std() + 1e-8)

        advantage_tensor = advantage_batch
        advantage_tensor = advantage_tensor.reshape(self.batch_size)
        # Normalize the advantages
        advantages = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)

        log_probability_tensor = log_probability_batch
        log_probability_tensor = log_probability_tensor.reshape(self.batch_size)
        
        with torch.no_grad():
            output = self.actor.forward(state_tensor)
            new_probabilities = Categorical(logits=output)
            new_log_probabilities = new_probabilities.log_prob(action_tensor)
        
        policy_ratios = torch.exp(new_log_probabilities - log_probability_tensor)

        advantage_times_ratio = advantages * policy_ratios

        g = advantages * torch.clamp(policy_ratios, 1 - self.clip, 1 + self.clip)

        loss = torch.min(advantage_times_ratio, g)

        values = self.critic.forward(state_tensor)
        mse_loss = nn.MSELoss()
        total_loss = -loss + 0.5 * mse_loss(values, returns)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.mean().backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()

    def save(self, msg=""):
        torch.save(self.actor.state_dict(), "./models/ppo_actor_model" + msg + ".pt")
        torch.save(self.critic.state_dict(), "./models/ppo_critic_model" + msg + ".pt")

    def load(self, msg=""):
        self.actor.load_state_dict(torch.load("./models/ppo_actor_model" + msg + ".pt"))
        self.actor.eval()
        self.critic.load_state_dict(torch.load("./models/ppo_critic_model" + msg + ".pt"))
        self.critic.eval()

    def get_action(self, state, reward, done):
        self.step_counter += 1

        if (self.step_counter % 1000000 == 0):
            self.save(msg="_" + str(self.step_counter))
    
        state_tensor = torch.tensor(state)

        state_tensor = state_tensor.reshape(1, 4, 84, 84)

        logits = self.actor.forward(state_tensor)

        probabilities = Categorical(logits=logits)

        value = self.critic.forward(state_tensor)

        action = probabilities.sample()

        log_p = probabilities.log_prob(action)

        self.rolloutbuffer.add_entry(state, action, reward, value, log_p, done)

        if done:
            self.rolloutbuffer.complete_trajectory(self.gamma);
            self.trajectory_complete = True

        if self.train_counter >= self.train_delay and self.trajectory_complete:
            self.train_counter = -1
            self.train()
        
        self.train_counter += 1

        return action.item()















