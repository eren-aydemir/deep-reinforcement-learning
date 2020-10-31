import gym

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random

from collections import namedtuple, deque
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

from dqn_agent import Agent

from unityagents import UnityEnvironment

class learnerDQN():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent, state_size=8, action_size=4,
                 seed=0, n_episodes=2000, max_t=1000,
                eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
        self.brain_name = self.env.brain_names[0]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        
        self.agent = agent
        self.scores = []
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        print("Initialized!")
        
    def train(self):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        
        print("Training!")
                
        scores = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start
        for i_episode in range(1, self.n_episodes+1):
            self.env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = self.env_info.vector_observations[0]
            score = 0
            for t in range(self.max_t):
                action = self.agent.act(state, eps).astype(int)
                self.env_info = self.env.step(action)[self.brain_name]
                next_state = self.env_info.vector_observations[0]
                reward = self.env_info.rewards[0]
                done = self.env_info.local_done[0]
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            self.scores.append(score)              # save most recent score
            eps = max(self.eps_end, self.eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        
    def load(self):
        # load the weights from file
        self.agent.qnetwork_local.load_state_dict(torch.load('checkpoint1.pth'))
        
    def plot(self):
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        