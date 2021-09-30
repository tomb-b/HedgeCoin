import torch
import numpy as np
from collections import namedtuple
from dqnModel import DoubleQLearningModel, ExperienceReplay, train_loop_ddqn
from marketv2 import *
from web3 import Web3, HTTPProvider


# Create the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = marketv2(device)

# Initializations
num_actions = env.action_space
num_states = env.observation_space

# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions)
ddqn.online_model.load_state_dict(torch.load("dqnCryptoInvestOnlineV2"))
ddqn.offline_model.load_state_dict(torch.load("dqnCryptoInvestOfflineV2"))