import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 0.001
LR_CRITIC = 0.001
WEIGHT_DECAY = 0

EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
EPSILON_DECAY = 0
LEARN_START = 0
UPDATE_EVERY = 10
UPDATES_PER_STEP = 10

class Agent:

    def __init__(self, config, state_size, action_size, num_agents, random_seed, device,
                 actor_trained_weight_filename=None,
                 critic_trained_weight_filename=None):

        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.device = device
        self.epsilon = EPSILON_MAX

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        if actor_trained_weight_filename:
            weights = torch.load(actor_trained_weight_filename)
            self.actor_local.load_state_dict(weights)
            self.actor_target.load_state_dict(weights)

        if critic_trained_weight_filename:
            weights = torch.load(critic_trained_weight_filename)
            self.critic_local.load_state_dict(weights)
            self.critic_target.load_state_dict(weights)


        self.memory = ReplayBuffer(action_size, config.buffer_size, config.mini_batch_size, random_seed, device)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done, self.num_agents)

        if len(self.memory) > LEARN_START:
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # Learn, if enough samples are available in memory
                if len(self.memory) > self.config.mini_batch_size:
                    for _ in range(UPDATES_PER_STEP):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA)

    def act(self, state, noise_weight=1.0, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()


        return np.clip(action, -1, 1)

    def reset(self):
        for i in range(self.num_agents):
            self.noise[i].reset()

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences


        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        self.epsilon = max(self.epsilon - EPSILON_DECAY, EPSILON_MIN)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed, device):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done, num_agents):
        for i in range(num_agents):
            e = self.experience(state[i], action[i], reward[i], next_state[i], done[i])
            self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)