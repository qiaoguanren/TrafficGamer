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

from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
from typing import Optional
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
from utils.dual_variable import DualVariable
import numpy as np
from tqdm import tqdm
from utils.utils import compute_advantage

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QRDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support):
        super(QRDQN, self).__init__()
        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        layer_init(self.linear1)
        layer_init(self.linear2)
        layer_init(self.V)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V
    
class IQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support):
        super(IQN, self).__init__()
        self.num_support = num_support

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # phi
        self.phi = nn.Linear(1, hidden_size[1], bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(hidden_size[1]), requires_grad = True)

        self.linear3 = nn.Linear(hidden_size[1], hidden_size[1])

        # Weight Init
        layer_init(self.linear1)
        layer_init(self.linear2)
        layer_init(self.V)
        layer_init(self.phi)
        layer_init(self.linear3)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # tau
        tau = torch.rand(self.num_support, 1).cuda()
        quants = torch.arange(0, self.num_support, 1.0).cuda()
        cos_trans = torch.cos(quants * tau * np.pi).unsqueeze(2) # (num_support, num_support, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        x = x.unsqueeze(1)
        x = x * rand_feat

        x = F.relu(self.linear3(x))
        # Output
        V = self.V(x).transpose(1,2) # (bs_size, 1, num_support)
        V = V.squeeze(1)
        return V, tau


class SplineDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support):
        super(SplineDQN, self).__init__()

        self.num_support = num_support
        self.K = num_support

        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (knots)
        self.V = nn.Linear(hidden_size[1], (3 * self.K - 1))

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        layer_init(self.linear1)
        layer_init(self.linear2)
        layer_init(self.V)
        layer_init(self.alpha)
        layer_init(self.beta)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        spline_param = self.V(x)
        scale_a = self.alpha(x)
        scale_a = torch.exp(scale_a)
        scale_b = self.beta(x)

        # split the last dimention to W, H, D
        W, H, D = torch.split(spline_param, self.K, dim=1)
        W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
        W = self.min_bin_width + (1 - self.min_bin_width * self.K) * W
        H = self.min_bin_height + (1 - self.min_bin_height * self.K) * H
        D = self.min_derivative + F.softplus(D)
        D = F.pad(D, pad=(1, 1))
        constant = np.log(np.exp(1 - 1e-3) - 1)
        D[..., 0] = constant
        D[..., -1] = constant

        # start and end x of each bin
        cumwidths = torch.cumsum(W, dim=-1).cuda()
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths[..., -1] = 1.0
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # (batch_sz, K)

        # start and end y of each bin
        cumheights = torch.cumsum(H, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (scale_a * cumheights + scale_b).cuda()
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # get bin index for each tau
        tau = torch.arange(0.5 * (1 / self.num_support), 1, 1 / self.num_support).cuda()
        tau = tau.expand((batch_size, self.num_support))

        cumwidths_expand = cumwidths.unsqueeze(dim=1).cuda()
        cumwidths_expand = cumwidths_expand.expand(-1, self.num_support, -1)  # (batch_sz, num_support, K+1)

        bin_idx = self.searchsorted_(cumwidths_expand, tau)

        # collect number
        input_cumwidths = cumwidths.gather(-1, bin_idx)
        input_bin_widths = widths.gather(-1, bin_idx)

        input_cumheights = cumheights.gather(-1, bin_idx)
        input_heights = heights.gather(-1, bin_idx)

        delta = heights / widths

        input_delta = delta.gather(-1, bin_idx)

        input_derivatives = D.cuda().gather(-1, bin_idx)
        input_derivatives_plus_one = D[..., 1:].cuda().gather(-1, bin_idx)

        # calculate quadratic spline for each tau
        theta = (tau - input_cumwidths) / input_bin_widths

        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
                    input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        return outputs

    def searchsorted_(self, bin_locations, inputs):
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

class NCQR(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support):
        super(NCQR, self).__init__()

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        layer_init(self.linear1)
        layer_init(self.linear2)
        layer_init(self.V)
        layer_init(self.alpha)
        layer_init(self.beta)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        quant = self.V(x)
        quant = torch.softmax(quant, dim=-1)
        quant = torch.cumsum(quant, dim=-1)

        # scale
        scale_a = F.relu(self.alpha(x))
        scale_b = self.beta(x)

        V = scale_a * quant + scale_b
        return V

class TrafficGamer(Constrainted_CCE_MAPPO):
    def __init__(self, state_dim: int, agent_number: int, config, device):
        super(TrafficGamer, self).__init__(state_dim, agent_number, config, device)
        
        self.N = config['N_quantile']
        self.cost_quantile = config['cost_quantile']
        self.tau_update = config['tau_update']
        self.LR_QN = config['LR_QN']
        self.type = config['type']
        self.method = config['method']
        self.input_action: bool = True
        if self.method == 'QRDQN':
            self.dis_build_QRDQN()
        elif self.method == 'IQN':
            self.dis_build_IQN()
        elif self.method == 'SplineDQN':
            self.dis_build_SplineDQN()
        elif self.method == 'NCQR':
            self.dis_build_NCQR()
        self.distributional_RL_optimizer = torch.optim.AdamW([
            {'params': self.cost_value_net_local.parameters(), 'lr': self.LR_QN, 'eps': 1e-5},
            {'params': self.cost_value_net_target.parameters(), 'lr': self.LR_QN, 'eps': 1e-5}])
            
    def dis_build_QRDQN(self) -> None:

        self.cost_value_net_local = QRDQN([256, 256], self.state_dim, self.N).to(self.device)
        self.cost_value_net_target = QRDQN([256, 256], self.state_dim, self.N).to(self.device)
        
        
    def dis_build_IQN(self) -> None:
        self.cost_value_net_local =  IQN([256, 256], self.state_dim, self.N).to(self.device)
        self.cost_value_net_target = IQN([256, 256], self.state_dim, self.N).to(self.device)

    def dis_build_SplineDQN(self) -> None:
        self.cost_value_net_local =  SplineDQN([256, 256], self.state_dim, self.N).to(self.device)
        self.cost_value_net_target =  SplineDQN([256, 256], self.state_dim, self.N).to(self.device)

    def dis_build_NCQR(self) -> None:
        self.cost_value_net_local =  NCQR([256, 256], self.state_dim, self.N).to(self.device)
        self.cost_value_net_target = NCQR([256, 256], self.state_dim, self.N).to(self.device)
        
    def quantile_regression_loss(self, expected, target, N):

        T_theta_tile = target.view(-1, N, 1).expand(-1, N, N).to(self.device)
        theta_a_tile = expected.view(-1, 1, N).expand(-1, N, N).to(self.device)
        quantile_tau = torch.arange(0.5 * (1 / N), 1, 1 / N).view(1, N).to(self.device)
        error_loss = T_theta_tile - theta_a_tile
        huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
        value_loss = (quantile_tau - (error_loss < 0).float()).abs() * huber_loss
        DQ_loss = value_loss.mean(dim=2).sum(dim=1).mean()
        return DQ_loss
    
    def update(self, transition, agent_index):

        logs = []
        
        
        states, observations, actions, rewards, costs, next_states, next_observations, dones, magnet = (
            self.sample(transition, agent_index)
        )

        dones = torch.stack(dones, dim=0).view(-1, 1)
        states = (
            torch.stack(states, dim=0)
            .reshape(-1, self.agent_number * self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        next_states = (
            torch.stack(next_states, dim=0)
            .reshape(-1, self.agent_number * self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        observations = (
            torch.stack(observations, dim=0)
            .reshape(-1, self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        next_observations = (
            torch.stack(next_observations, dim=0)
            .reshape(-1, self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )

        rewards = torch.stack(rewards, dim=0).to(self.device)
        costs = torch.stack(costs, dim=0).to(self.device)

        actions = torch.stack(actions, dim=0).to(self.device)
        
        for _ in range(self.epochs*5):
                
            nominal_data_batch = observations

            m_loss, _ = self.density_model.log_probs(inputs=nominal_data_batch,
                                                            cond_inputs=None)
            self.density_optimizer.zero_grad()
            density_loss = -m_loss.mean()
            density_loss.backward()
            self.density_optimizer.step()

        with torch.no_grad():
            
            # get ratio
            old_policy = self.get_action_dist(observations)
            old_log_probs = old_policy.log_prob(actions)
            
            if  self.magnet:
                magnet_signal = torch.tensor(
                    [magnet[i].log_prob(actions[i]) for i in range(len(actions))]
                )
                rewards = rewards + self.eta_coef1 * magnet_signal[:, None].to(self.device)
            
            next_state_value = self.value(next_states)

            _, log_prob_game = self.density_model.log_probs(inputs=next_observations,
                                                                        cond_inputs=None)
            
            log_prob_game = F.sigmoid(((log_prob_game - log_prob_game.mean()) / log_prob_game.std()).exp())
            beta_t = self.beta_coef / log_prob_game

            td_target = rewards + self.gamma * next_state_value * (1 - dones) + beta_t
        
            td_value = self.value(states)
            td_delta = td_target - td_value

            # calculate GAE
            reward_advantages = compute_advantage(
                self.gae, td_delta, self.device, dones, self.gamma, self.lamda)
            # advantage_normalization
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (
                reward_advantages.std() + 1e-8
            )
            
            if self.method == 'QRDQN' or self.method=='SplineDQN' or self.method=='NCQR':
                distributional_cost_values = self.cost_value_net_local(observations)
                distributional_cost_values_targets_next = self.cost_value_net_local(next_observations)
                distributional_cost_values_targets = costs + \
                            (self.gamma * distributional_cost_values_targets_next.to(self.device) * (1 - dones))
            elif self.method == 'IQN':
                distributional_cost_values, tau = self.cost_value_net_local(observations)
                distributional_cost_values_targets_next, tau_next = self.cost_value_net_local(next_observations)
                distributional_cost_values_targets = costs + \
                            (self.gamma * distributional_cost_values_targets_next.to(self.device) * (1 - dones))
            
            if self.type == 'VaR':
                # Caculate the cost values using VaR method
                cost_values = distributional_cost_values[:,self.cost_quantile-1].view(distributional_cost_values.shape[0], 1)
                cost_values_target = distributional_cost_values_targets_next[:,self.cost_quantile-1].view(distributional_cost_values_targets_next.shape[0], 1)
            elif self.type == 'CVaR':
                # Caculate the cost values using CVaR method
                VaR = distributional_cost_values[:, self.cost_quantile-1].view(distributional_cost_values.shape[0], 1)
                alpha = self.cost_quantile / self.N
                exp = torch.mean(abs(distributional_cost_values - VaR), dim=1).view(distributional_cost_values.shape[0], 1)
                cost_values = VaR + exp / (1.0 - alpha)
                
                VaR_t = distributional_cost_values_targets[:, self.cost_quantile-1].view(distributional_cost_values_targets.shape[0], 1)
                exp_t = torch.mean(abs(distributional_cost_values_targets - VaR_t), dim=1).view(distributional_cost_values_targets.shape[0], 1)
                cost_values_target = VaR_t + exp_t / (1.0 - alpha)
                
            cost_advantages = costs + self.gamma * cost_values_target * (1 - dones) - cost_values
            cost_advantages = (cost_advantages - cost_advantages.mean())
            returns = reward_advantages + td_value
            
        for epoch in tqdm(range(self.epochs)):
            log = {}
            log["magnet_signal"] = magnet_signal.mean().item()
            if self.method == 'QRDQN' or self.method=='SplineDQN' or self.method=='NCQR':
                    
                distributional_cost_values_expected = self.cost_value_net_local(observations)
                DQ_loss = self.quantile_regression_loss(distributional_cost_values_expected, distributional_cost_values_targets, self.N)
            elif self.method == 'IQN':
                    
                distributional_cost_values_expected = self.cost_value_net_local(observations)
                DQ_loss = self.quantile_regression_loss(distributional_cost_values_expected, distributional_cost_values_targets, self.N)
            log["DQ_loss"] = DQ_loss.item()
            
            # for target_param, local_param in zip(self.cost_value_net_target.parameters(), self.cost_value_net_local.parameters()):
            #         target_param.data.copy_(
            #             self.tau_update * local_param.data + (1.0 - self.tau_update) * target_param.data)
            
            new_policy = self.get_action_dist(observations)
            log_probs = new_policy.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # clipping
            ratio = ratio.flatten(start_dim=1)
            surr1 = ratio * reward_advantages
            surr2 = reward_advantages * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)

            value = self.value(states)
            # cost_value = self.cost_value(states)
            # log["cost_value"] = cost_value.mean().item()

            # pi and value loss
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(value, td_target.detach()))
            # cost_value_loss = torch.mean(F.mse_loss(cost_td_delta, cost_values.detach()))
            entropy = new_policy.entropy()

            log["entropy"] = entropy.mean().item()

            current_penalty = self.dual.nu().item()
            log["distance_current_penalty"] = current_penalty
            pi_loss += current_penalty * torch.mean(cost_advantages * ratio)
            pi_loss /= (1 + current_penalty)

            total_loss = (
                pi_loss + 0.5 * value_loss + 0.5 * DQ_loss  - entropy * self.entropy_coef
            ).mean()
            log["total_loss"] = total_loss.mean().item()
            self.optimizer.zero_grad()
            self.distributional_RL_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.cost_value_net_local.parameters(), 0.5)
            self.optimizer.step()
            self.distributional_RL_optimizer.step()
            logs.append(log)
            # approx_kl_div = torch.mean(old_log_probs - log_probs).detach().cpu().numpy()
            # if approx_kl_div > 1.5 * self.target_kl:
            #     break

        average_cost = np.mean(costs.cpu().numpy())
        self.dual.update_parameter(average_cost)
        return logs