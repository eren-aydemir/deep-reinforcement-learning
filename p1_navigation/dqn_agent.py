import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from replay_buffer import ReplayBuffer

class Agent():

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64],
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3,
                 learning_rate=5e-4, update_every=4):
        """ state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hidden_layers (list of int ; optional): number of each layer nodes
            buffer_size (int ; optional): replay buffer size
            batch_size (int; optional): minibatch size
            gamma (float; optional): discount factor
            tau (float; optional): for soft update of target parameters
            learning_rate (float; optional): learning rate
            update_every (int; optional): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = learning_rate
        self.update_every = update_every

        # detect GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size,
                                   self.batch_size, seed, self.device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """ state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
            states, actions, rewards, next_states, dones = experiences

            # Calculate target value
            self.qnetwork_target.eval()
            with torch.no_grad():
                Q_dash_local = self.qnetwork_local(next_states)
                Q_dash_target = self.qnetwork_target(next_states)
                argmax_action = torch.max(Q_dash_local, dim=1, keepdim=True)[1]
                Q_dash_max = Q_dash_target.gather(1, argmax_action)
                y = rewards + gamma * Q_dash_max * (1 - dones)
            self.qnetwork_target.train()

            # Predict Q-value
            self.optimizer.zero_grad()
            Q = self.qnetwork_local(states)
            y_pred = Q.gather(1, actions)

            # TD-error
            loss = torch.sum((y - y_pred)**2)

            # Optimize
            loss.backward()
            self.optimizer.step()

            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)