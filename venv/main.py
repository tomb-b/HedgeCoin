import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from market import *
from ppoModel import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = market.market(device)
state_size = env.observation_space
num_actions = env.action_space
model = PPOModel(state_size, num_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
print(model)
model.load_state_dict(torch.load("ppoCryptoInvestAllTest"))

max_episodes, save_time = 400, 100
R, R_avg = [], []
prev_time = time.time()

for episode in range(max_episodes):
    model.run_env(env, optimizer, model, device)
    R.append(model.eval_model(env, model, device))
    R_avg.append(.05 * R[-1] + .95 * R_avg[episode - 1]) if episode > 0 else R_avg.append(R[-1])

    print('Return: (' + str(episode) + '): ' + str(R[-1]))
    print('Running average (' + str(episode) + '): ' + str(R_avg[-1]))
    env.render()

    if R_avg[-1] > 50000 and episode > 40:
        print('Congratulations, problem solved!')
        break
    elif (time.time() - prev_time) > save_time:
        print('Checkpointing model...')
        torch.save(model.state_dict(), "ppoCryptoInvestAllTest")
        prev_time = time.time()

print('Saving model...')
torch.save(model.state_dict(), "ppoCryptoInvestAllTest")