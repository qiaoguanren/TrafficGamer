# BSD 3-Clause License

# Copyright (c) 2024, Guanren Qiao

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from tqdm import tqdm

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Policy, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 2 * 2), 0.01),
        )

    def forward(self, state):
        result = self.actor(state)
        mean = result[..., :2]

        b = result[..., 2:]
        b = F.elu_(b, alpha=1.0) + 1.1
        return mean, b


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, agent_number):
        super(ValueNet, self).__init__()

        self.f = nn.Sequential(
            layer_init(nn.Linear(state_dim * agent_number, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, states):

        v = self.f(states)
        return v
    
class PPO:
    def __init__(self, 
                 state_dim: int, 
                 agent_number: int,
                 config,
                 device):
        self.hidden_dim =config['hidden_dim']
        self.state_dim = state_dim
        # self.old_value = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_lr = config['actor_learning_rate']
        self.critic_lr = config['critic_learning_rate']
        self.cost_critic_lr = config['constrainted_critic_learning_rate']
        self.density_lr = config['density_learning_rate']
        self.lamda = config['lamda'] #discount factor
        self.eps = config['eps'] #clipping parameter
        self.gamma = config['gamma'] # the factor of caculating GAE
        self.device = device
        self.agent_number = agent_number
        self.offset = config['offset']
        self.entropy_coef = config['entropy_coef']
        self.epochs = config['epochs']
        self.algorithm = config['algorithm']
        self.batch_size = config['batch_size']
        self.gae = config['gae']
        self.target_kl = config['target_kl']

        self.pi = Policy(state_dim, self.hidden_dim).to(device)
        self.old_pi = Policy(state_dim, self.hidden_dim).to(device)
        self.value = ValueNet(state_dim, self.hidden_dim, self.agent_number).to(device)
        self.cost_value = ValueNet(state_dim, self.hidden_dim, self.agent_number).to(device)

        params = [
            {'params': self.pi.parameters(), 'lr': self.actor_lr, 'eps': 1e-5},
            {'params': self.value.parameters(), 'lr': self.critic_lr, 'eps': 1e-5},
            {'params': self.cost_value.parameters(), 'lr': self.cost_critic_lr, 'eps': 1e-5}
        ]

        self.optimizer = torch.optim.AdamW(params)

        
    def sample(self, transition, choose_index):

        s = []
        a = []
        r = []
        s_next = []
        done = []

        with torch.no_grad():
            for row in range(0, self.batch_size):
                s += transition[row]['observations'][choose_index]
                a += transition[row]['actions'][choose_index]
                r += transition[row]['rewards'][choose_index]
                s_next += transition[row]['next_observations'][choose_index]
                done += transition[row]['dones']

        return s, a, r, s_next, done

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.flatten(state,start_dim=0).unsqueeze(0)
            mean, var = self.pi(state)
            action = Laplace(mean, var)
            action = action.sample()
        return action

    
    def update(self, transition,choose_index):
        logs = []
        for epoch in tqdm(range(self.epochs)):
                log = {}
                states,  actions, rewards, next_states, dones= self.sample(transition,choose_index)
                states = torch.stack(states, dim=0).flatten(start_dim=1)
                next_states = torch.stack(next_states, dim=0).flatten(start_dim=1)
                rewards = torch.stack(rewards, dim=0).view(-1,1)
                dones = torch.stack(dones, dim=0).view(-1,1)
                actions = torch.stack(actions, dim=0).flatten(start_dim=1)

                # td_error
                with torch.no_grad():
                    next_state_value = self.value(next_states)
                    td_target = rewards + self.gamma * next_state_value * (1-dones)
                    # _, log_prob_game = self.density_model.log_probs(inputs=next_states,
                    #                                                 cond_inputs=None)
                    # log_prob_game = F.sigmoid((log_prob_game - log_prob_game.mean()) / log_prob_game.std())
                    # beta_t = self.beta_coef / (log_prob_game)
                    
                    # td_target = td_target + beta_t
                    td_value = self.value(states)

                    td_delta = td_target - td_value
                
                    # calculate GAE
                    advantage = 0
                    advantage_list = []
                    td_delta = td_delta.cpu().detach().numpy()
                    for delta in td_delta[::-1]:
                        advantage = self.gamma * self.lamda * advantage + delta
                        advantage_list.append(advantage)
                    advantage_list.reverse()
                    advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device).reshape(-1,1)
                    #advantage_normalization
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    log["advantage"] = advantage.mean().item()
            
                    # get ratio
                    mean, b = self.pi(states)
                    old_policy = Laplace(mean, b)
                    old_log_probs = old_policy.log_prob(actions)

                mean, b = self.pi(states)
                new_policy = Laplace(mean, b)
                log_probs = new_policy.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
        
                # clipping
                ratio = ratio.flatten(start_dim=1)
                surr1 = ratio * advantage
                surr2 = advantage * torch.clamp(ratio, 1-self.eps, 1+self.eps)

                value = self.value(states)
                
                # pi and value loss
                pi_loss = torch.mean(-torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(value, td_target.detach()))

                total_loss = (pi_loss + 0.5*value_loss - new_policy.entropy() * self.entropy_coef)
                # total_loss = pi_loss + value_loss
                # entropy_list.append(new_policy.entropy().mean().item())
                log["total_loss"] = total_loss.mean().item()

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
                self.optimizer.step()

                logs.append(log)
        return logs
        
