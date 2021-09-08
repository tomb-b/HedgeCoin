# Import dependencies
import torch
import numpy as np
import gym
from collections import namedtuple
from dqnModel import DoubleQLearningModel, ExperienceReplay, train_loop_ddqn
from marketv2 import *


# Create the environment
#env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = marketv2(device)

# Enable visualization? Does not work in all environments.
enable_visualization = False

# Initializations
#num_actions = env.action_space.n
#num_states = env.observation_space.shape[0]
num_actions = env.action_space
num_states = env.observation_space
num_episodes = 1200
batch_size = 128
gamma = .94
learning_rate = 1e-4

# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
# for training
replay_buffer = ExperienceReplay(device, num_states)

# Train
R, R_avg = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma)