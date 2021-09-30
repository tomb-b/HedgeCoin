import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


gamma = 0.99
adv_lambda = 0.97
eps = 0.2
c1 = 0.5
c2 = 0.002
epochs = 60
batch_size = 8
exploration_fac = 0.2

class PPOModel(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        
        self.critic_net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
        self.actor_net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Tanh())
        
    def forward(self, state):
        val = self.critic_net(state)
        mean = self.actor_net(state)
        dist = torch.distributions.Normal(mean, exploration_fac)
        
        return dist, val
        

    def get_returns(self, rewards, values, terminal, next_val):
        adv_estimate = 0.
        returns = []
        values_ext = [*values, next_val]

        for step in reversed(range(len(values))):
            delta = rewards[step] + gamma * values_ext[step + 1] * terminal[step] - values_ext[step]
            adv_estimate = delta + gamma * adv_lambda * adv_estimate * terminal[step]
            returns.insert(0, adv_estimate + values_ext[step])

        return torch.FloatTensor(returns)

    def run_env(self, env, optimizer, model, device):
        actions, states, values, log_probs, rewards, terminal = [], [], [], [], [], []
        state = torch.FloatTensor(env.reset()).to(device)
        done = False
        num_steps = 0

        while not done:
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action)
            num_steps += 1
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(np.array([reward])).to(device))
            states.append(state)
            actions.append(action)
            terminal.append(0. if done else 1.)
            state = torch.FloatTensor(next_state).to(device)

        _, next_val = model(torch.FloatTensor(state).to(device))
        states = torch.squeeze(torch.stack(states).to(device), 1)
        actions = torch.squeeze(torch.stack(actions).to(device), 1)
        returns = self.get_returns(rewards, values, terminal, next_val)
        log_probs = torch.FloatTensor(log_probs).to(device)
        values = torch.FloatTensor(values)
        advantages = returns - values
        self.update_model(env, optimizer, model, device, states, actions, log_probs, returns, advantages)

        return num_steps

    def update_model(self, env, optimizer, model, device, states, actions, log_probs, returns, advantages):
        steps, acc_loss = 0, 0.
        dataset = TensorDataset(states, actions, log_probs, returns, advantages)

        for epoch in range(epochs):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for b_states, b_actions, b_log_probs, b_returns, b_advantages in loader:
                dist, values = model(b_states.to(device))
                new_log_probs = dist.log_prob(b_actions)
                ratio = (new_log_probs - b_log_probs).exp()
                loss_actor_net = -torch.min(ratio * b_advantages,
                                            torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantages)
                loss_critic_net = (returns - values).pow(2)
                loss_ppo = loss_actor_net.mean() + c1 * loss_critic_net.mean() - c2 * dist.entropy().mean()

                optimizer.zero_grad()
                loss_ppo.backward()
                optimizer.step()

                steps += 1
                acc_loss += loss_ppo.item()

        print('Mean loss for all epochs: ' + str(acc_loss / steps))

    def eval_model(self, env, model, device):
        done = False
        total_reward = 0
        state = env.reset()

        while not done:
            dist, val = model(torch.FloatTensor(state).detach().to(device))
            action = dist.mean.detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward
