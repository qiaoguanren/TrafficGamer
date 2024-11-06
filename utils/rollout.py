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

from typing import List
import matplotlib.pyplot as plt
from utils.utils import (
    add_new_agent,
    cost_function,
    reward_function,
    seed_everything,
)
from neptune.types import File
from PIL import Image as img
import torch,io
import torch.nn.functional as F
from torch.distributions import Laplace
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path
from algorithm.mappo import MAPPO
from algorithm.ppo import PPO
from utils.utils import get_auto_pred, get_transform_mat, generate_tmp_gif_path, get_v_transform_mat
from visualization.vis import plot_traj_with_data

def PPO_process_batch(
        args,
        batch,
        new_input_data,
        model,
        agents: List[MAPPO],
        choose_agent,
        offset,
        scenario_static_map,
        scenario_num,
        transition_list,


        render,
        agent_num,
        dataset_type='av2'
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        if model.output_head:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["loc_propose_head"],
                    pred["scale_propose_pos"][..., : model.output_dim],
                    pred["conc_propose_head"],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["loc_refine_head"],
                    pred["scale_refine_pos"][..., : model.output_dim],
                    pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

        auto_pred = pred

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )
        transformed_v = torch.einsum("bi,bij->bj", init_v, init_rot_mat)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred["pi"]
        pi_eval = F.softmax(pi, dim=-1)

        state_temp_list = []
        global_state = pred["first_m"]
        for i in range(agent_num):
            state_temp_list.append(global_state[choose_agent[i]])
        frames = []
        pred_position = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)
        pred_heading = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
        pred_velocity = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)

        for timestep in range(0, model.num_future_steps, offset):

            best_mode = pi_eval.argmax(dim=-1)

            magnet_list = []
            action_list = []
            true_trans_position_refine=new_true_trans_position_refine

            for i in range(agent_num):
                pred_v = (
                    pred["loc_refine_pos"][choose_agent[i], :, 1 : offset + 2]
                    - pred["loc_refine_pos"][choose_agent[i], :, : offset + 1]
                ) / 0.1
                a = (
                    pred_v[:, 1 : offset + 1].norm(dim=-1)
                    - pred_v[:, :offset].norm(dim=-1)
                ) / 0.1
                kappa = (
                    pred["loc_refine_head"][choose_agent[i], :, 1 : offset + 1, 0]
                    - pred["loc_refine_head"][choose_agent[i], :, :offset, 0]
                ) / (pred_v[:, :offset].norm(dim=-1) * 0.1 + 1 / 2 * (0.1**2) * a)
                a = a.mean(-1)
                kappa = kappa.mean(-1)
                action = agents[i].choose_action(state_temp_list[i].flatten()[None, :])[
                    0
                ]
                action_a = action[0].clip(-1,1) * 5
                action_kappa = action[1].clip(-1,1) * 0.05

                v = transformed_v[choose_agent[i]]
                newpos = torch.zeros(offset + 2, 2)
                newhead = torch.zeros(offset + 2)
                for t in range(offset + 1):
                    newpos[t + 1, 0] = (
                        newpos[t, 0]
                        + v[0] * 0.1
                        + 1 / 2 * action_a * torch.cos(newhead[t]) * (0.1) ** 2
                    )
                    newpos[t + 1, 1] = (
                        newpos[t, 1]
                        + v[1] * 0.1
                        + 1 / 2 * action_a * torch.sin(newhead[t]) * (0.1) ** 2
                    )
                    newhead[t + 1] = newhead[t] + action_kappa * (
                        v.norm() * 0.1 + 1 / 2 * action_a * (0.1) ** 2
                    )
                    v_abs = v.norm() + action_a * 0.1
                    v[0] = v_abs * torch.cos(newhead[t + 1])
                    v[1] = v_abs * torch.sin(newhead[t + 1])

                auto_pred["loc_refine_pos"][
                    choose_agent[i],
                    best_mode[choose_agent[i]],
                    : offset + 1,
                    : model.output_dim,
                ] = newpos[1:]
                auto_pred["loc_refine_head"][
                    choose_agent[i], best_mode[choose_agent[i]], : offset + 1
                ] = newhead[1:].unsqueeze(-1)

                action_list.append(action)
                mix = torch.distributions.Categorical(pi_eval[choose_agent[i]])
                data_mean = torch.stack(
                    [
                        (a / 5).clip(-1 + 1e-4, 1 - 1e-4),
                        (kappa / 0.05).clip(-1 + 1e-4, 1 - 1e-4),
                    ],
                    -1,
                )
                data_scale = torch.ones_like(data_mean)
                data_scale[..., 0] = 0.1 # adjust these coeffs
                data_scale[..., 1] = 0.01 #

                comp = torch.distributions.Independent(
                    torch.distributions.Laplace(data_mean, data_scale), 1
                )
                gmm = torch.distributions.MixtureSameFamily(mix, comp)
                magnet_list.append(gmm)

            (
                new_data,
                auto_pred,
                _,
                _,
                (new_true_trans_position_propose, new_true_trans_position_refine),
                (traj_propose, traj_refine),
            ) = get_auto_pred(
                new_data,
                model,
                auto_pred["loc_refine_pos"][
                    torch.arange(traj_propose.size(0)), best_mode
                ],
                auto_pred["loc_refine_head"][
                    torch.arange(traj_propose.size(0)), best_mode, :, 0
                ],
                offset,
                anchor=(init_origin, init_theta, init_rot_mat),
            )

            next_state_temp_list = []
            global_next_state = auto_pred["first_m"]
            for i in range(agent_num):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
            for i in range(agent_num):
                transition_list[batch]["observations"][i].append(state_temp_list[i])
                transition_list[batch]["actions"][i].append(action_list[i])
                transition_list[batch]["next_observations"][i].append(next_state_temp_list[i])
                transition_list[batch]["magnet"][i].append(magnet_list[i])
            if timestep == model.num_future_steps - offset:
                transition_list[batch]["dones"].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]["dones"].append(torch.tensor(0).cuda())
            for i in range(agent_num):
                reward = reward_function(
                    new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map, dataset_type=dataset_type
                )
                transition_list[batch]["rewards"][i].append(
                    torch.tensor([reward]).cuda()
                )
                cost = cost_function(new_data.clone(), model, choose_agent[i], args.distance_limit, choose_agent)
                transition_list[batch]["costs"][i].append(torch.tensor([cost]).cuda())

            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred["pi"], dim=-1)
            if render:
                for t in range(offset):
                    plot_traj_with_data(
                        new_data,
                        scenario_static_map,
                        agent_number=agent_num,
                        scenario_num=scenario_num,
                        bounds=50,
                        t=11 - offset + t,
                        dataset_type=dataset_type,
                        choose_agent=choose_agent
                    )
                    # for agent in range(agent_num):
                    #     for j in range(6):
                    #         xy = true_trans_position_refine[choose_agent[agent]].cpu()
                    #         plt.plot(xy[j, ..., 0], xy[j, ..., 1])
                    # plot_destination(args.scenario)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    frame = img.open(buf)
                    frames.append(frame)

                pred_position[:,timestep:timestep+offset,:] = new_data['agent']['position'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
                pred_heading[:,timestep:timestep+offset] = new_data['agent']['heading'][:,model.num_historical_steps-offset:model.num_historical_steps]
                pred_velocity[:,timestep:timestep+offset,:] = new_data['agent']['velocity'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
            
        if render:
            import imageio

            # Specify the file path for the GIF
            gif_path = generate_tmp_gif_path()
            tmp = imageio.mimsave(gif_path, frames, fps=10)
            # run["val/rollout"].append(File(gif_path))
            if args.track:
                wandb.log({"val/rollout": wandb.Video(gif_path, format="gif")})
        
            return pred_position, pred_heading, pred_velocity
        
def expert_process_batch(
        args,
        batch,
        new_input_data,
        model,
        agents: List[MAPPO],
        choose_agent,
        offset,
        scenario_static_map,
        run,
        transition_list,
        render,
        agent_num,
        current_agent_index
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        if model.output_head:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["loc_propose_head"],
                    pred["scale_propose_pos"][..., : model.output_dim],
                    pred["conc_propose_head"],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["loc_refine_head"],
                    pred["scale_refine_pos"][..., : model.output_dim],
                    pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

        auto_pred = pred

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )
        transformed_v = torch.einsum("bi,bij->bj", init_v, init_rot_mat)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred["pi"]
        pi_eval = F.softmax(pi, dim=-1)

        state_temp_list = []
        global_state = pred["first_m"]
        for i in range(agent_num):
            state_temp_list.append(global_state[choose_agent[i]])

        for timestep in range(0, model.num_future_steps, offset):

            best_mode = pi_eval.argmax(dim=-1)
            reg_mask_list = []
            for i in range(agent_num):
                reg_mask = new_data['agent']['predict_mask'][choose_agent[i], model.num_historical_steps:]
                reg_mask_list.append(reg_mask)

            magnet_list = []
            action_list = []
            true_trans_position_refine=new_true_trans_position_refine

            sample_action_list = []
            for i in range(agent_num):
                pred_v = (
                    pred["loc_refine_pos"][choose_agent[i], :, 1 : offset + 2]
                    - pred["loc_refine_pos"][choose_agent[i], :, : offset + 1]
                ) / 0.1
                a = (
                    pred_v[:, 1 : offset + 1].norm(dim=-1)
                    - pred_v[:, :offset].norm(dim=-1)
                ) / 0.1
                kappa = (
                    pred["loc_refine_head"][choose_agent[i], :, 1 : offset + 1, 0]
                    - pred["loc_refine_head"][choose_agent[i], :, :offset, 0]
                ) / (pred_v[:, :offset].norm(dim=-1) * 0.1 + 1 / 2 * (0.1**2) * a)
                a = a.mean(-1)
                kappa = kappa.mean(-1)
                action = agents[i].choose_action(state_temp_list[i].flatten()[None, :])[
                    0
                ]
                action_a = action[0].clip(-1,1) * 5
                action_kappa = action[1].clip(-1,1) * 0.05
                v = transformed_v[choose_agent[i]]
                newpos = torch.zeros(offset + 2, 2)
                newhead = torch.zeros(offset + 2)
                for t in range(offset + 1):
                    newpos[t + 1, 0] = (
                        newpos[t, 0]
                        + v[0] * 0.1
                        + 1 / 2 * action_a * torch.cos(newhead[t]) * (0.1) ** 2
                    )
                    newpos[t + 1, 1] = (
                        newpos[t, 1]
                        + v[1] * 0.1
                        + 1 / 2 * action_a * torch.sin(newhead[t]) * (0.1) ** 2
                    )
                    newhead[t + 1] = newhead[t] + action_kappa * (
                        v.norm() * 0.1 + 1 / 2 * action_a * (0.1) ** 2
                    )
                    v_abs = v.norm() + action_a * 0.1
                    v[0] = v_abs * torch.cos(newhead[t + 1])
                    v[1] = v_abs * torch.sin(newhead[t + 1])
                    
                sample_action_list.append(newpos.cuda())

                action_list.append(action)
                mix = torch.distributions.Categorical(pi_eval[choose_agent[i]])
                data_mean = torch.stack(
                    [
                        (a / 5).clip(-1 + 1e-4, 1 - 1e-4),
                        (kappa / 0.05).clip(-1 + 1e-4, 1 - 1e-4),
                    ],
                    -1,
                )
                data_scale = torch.ones_like(data_mean)
                data_scale[..., 0] = 0.1 # adjust these coeffs
                data_scale[..., 1] = 0.01 #

                comp = torch.distributions.Independent(
                    torch.distributions.Laplace(data_mean, data_scale), 1
                )
                gmm = torch.distributions.MixtureSameFamily(mix, comp)
                magnet_list.append(gmm)
                
            for i in range(agent_num):
                if i==current_agent_index:
                    best_mode[choose_agent[i]] = 5 # sometimes 0 is better if you find expert reward is lower than real rewards.
                else:
                    l2_norm = (torch.norm(auto_pred['loc_refine_pos'][choose_agent[i],:,:offset, :2] -
                                        sample_action_list[i][1:6], p=2, dim=-1) * reg_mask_list[i][timestep:timestep+offset].unsqueeze(0)).sum(dim=-1)
                    action_suggest_index=l2_norm.argmin(dim=-1)
                    best_mode[choose_agent[i]] = action_suggest_index

            (
                new_data,
                auto_pred,
                _,
                _,
                (new_true_trans_position_propose, new_true_trans_position_refine),
                (traj_propose, traj_refine),
            ) = get_auto_pred(
                new_data,
                model,
                auto_pred["loc_refine_pos"][
                    torch.arange(traj_propose.size(0)), best_mode
                ],
                auto_pred["loc_refine_head"][
                    torch.arange(traj_propose.size(0)), best_mode, :, 0
                ],
                offset,
                anchor=(init_origin, init_theta, init_rot_mat),
            )

            next_state_temp_list = []
            global_next_state = auto_pred["first_m"]
            for i in range(agent_num):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
                if i==current_agent_index:

                    reward = reward_function(
                        new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map
                    )
                    transition_list[batch]["rewards"].append(
                        torch.tensor([reward]).cuda()
                    )

            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred["pi"], dim=-1) 
                       
def qcnet_baseline_process_batch(
        new_input_data,
        model,
        choose_agent,
        offset,
        scenario_static_map,
        transition_list,
        threeD,
        agent_num
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        if model.output_head:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["loc_propose_head"],
                    pred["scale_propose_pos"][..., : model.output_dim],
                    pred["conc_propose_head"],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["loc_refine_head"],
                    pred["scale_refine_pos"][..., : model.output_dim],
                    pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

        auto_pred = pred

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )
        transformed_v = torch.einsum("bi,bij->bj", init_v, init_rot_mat)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred["pi"]
        pi_eval = F.softmax(pi, dim=-1)

        state_temp_list = []
        global_state = pred["first_m"]
        for i in range(agent_num):
            state_temp_list.append(global_state[choose_agent[i]])
            
        pred_position = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)
        pred_heading = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
        pred_velocity = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)

        for timestep in range(0, model.num_future_steps, offset):
            
            best_mode = pi_eval.argmax(dim=-1)

            (
                new_data,
                auto_pred,
                _,
                _,
                (new_true_trans_position_propose, new_true_trans_position_refine),
                (traj_propose, traj_refine),
            ) = get_auto_pred(
                new_data,
                model,
                auto_pred["loc_refine_pos"][
                    torch.arange(traj_propose.size(0)), best_mode
                ],
                auto_pred["loc_refine_head"][
                    torch.arange(traj_propose.size(0)), best_mode, :, 0
                ],
                offset,
                anchor=(init_origin, init_theta, init_rot_mat),
            )

            next_state_temp_list = []
            global_next_state = auto_pred["first_m"]
            for i in range(agent_num):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
           
            for i in range(agent_num):
                reward = reward_function(
                    new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map)
                transition_list["rewards"][i].append(
                    torch.tensor([reward]).cuda()
                )

            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred["pi"], dim=-1)
            
        if threeD:
            pred_position[:,timestep:timestep+offset,:] = new_data['agent']['position'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
            pred_heading[:,timestep:timestep+offset] = new_data['agent']['heading'][:,model.num_historical_steps-offset:model.num_historical_steps]
            pred_velocity[:,timestep:timestep+offset,:] = new_data['agent']['velocity'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
            
            return pred_position, pred_heading, pred_velocity
      
def PPO_process_batch_confined_actions(
        args,
        batch,
        new_input_data,
        model,
        agents: List[MAPPO],
        choose_agent,
        offset,
        scenario_static_map,
        scenario_num,
        transition_list,
        render,
        agent_num,
        dataset_type='av2'
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        if model.output_head:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["loc_propose_head"],
                    pred["scale_propose_pos"][..., : model.output_dim],
                    pred["conc_propose_head"],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["loc_refine_head"],
                    pred["scale_refine_pos"][..., : model.output_dim],
                    pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

        auto_pred = pred

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )
        transformed_v = torch.einsum("bi,bij->bj", init_v, init_rot_mat)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred["pi"]
        pi_eval = F.softmax(pi, dim=-1)

        state_temp_list = []
        global_state = pred["first_m"]
        for i in range(agent_num):
            state_temp_list.append(global_state[choose_agent[i]])

        for timestep in range(0, model.num_future_steps, offset):

            best_mode = pi_eval.argmax(dim=-1)
            reg_mask_list = []
            for i in range(agent_num):
                reg_mask = new_data['agent']['predict_mask'][choose_agent[i], model.num_historical_steps:]
                reg_mask_list.append(reg_mask)

            magnet_list = []
            action_list = []
            true_trans_position_refine=new_true_trans_position_refine

            sample_action_list = []
            for i in range(agent_num):
                pred_v = (
                    pred["loc_refine_pos"][choose_agent[i], :, 1 : offset + 2]
                    - pred["loc_refine_pos"][choose_agent[i], :, : offset + 1]
                ) / 0.1
                a = (
                    pred_v[:, 1 : offset + 1].norm(dim=-1)
                    - pred_v[:, :offset].norm(dim=-1)
                ) / 0.1
                kappa = (
                    pred["loc_refine_head"][choose_agent[i], :, 1 : offset + 1, 0]
                    - pred["loc_refine_head"][choose_agent[i], :, :offset, 0]
                ) / (pred_v[:, :offset].norm(dim=-1) * 0.1 + 1 / 2 * (0.1**2) * a)
                a = a.mean(-1)
                kappa = kappa.mean(-1)
                action = agents[i].choose_action(state_temp_list[i].flatten()[None, :])[
                    0
                ]
                action_a = action[0].clip(-1,1) * 5
                action_kappa = action[1].clip(-1,1) * 0.05
                v = transformed_v[choose_agent[i]]
                newpos = torch.zeros(offset + 2, 2)
                newhead = torch.zeros(offset + 2)
                for t in range(offset + 1):
                    newpos[t + 1, 0] = (
                        newpos[t, 0]
                        + v[0] * 0.1
                        + 1 / 2 * action_a * torch.cos(newhead[t]) * (0.1) ** 2
                    )
                    newpos[t + 1, 1] = (
                        newpos[t, 1]
                        + v[1] * 0.1
                        + 1 / 2 * action_a * torch.sin(newhead[t]) * (0.1) ** 2
                    )
                    newhead[t + 1] = newhead[t] + action_kappa * (
                        v.norm() * 0.1 + 1 / 2 * action_a * (0.1) ** 2
                    )
                    v_abs = v.norm() + action_a * 0.1
                    v[0] = v_abs * torch.cos(newhead[t + 1])
                    v[1] = v_abs * torch.sin(newhead[t + 1])
                    
                sample_action_list.append(newpos.cuda())

                action_list.append(action)
                mix = torch.distributions.Categorical(pi_eval[choose_agent[i]])
                data_mean = torch.stack(
                    [
                        (a / 5).clip(-1 + 1e-4, 1 - 1e-4),
                        (kappa / 0.05).clip(-1 + 1e-4, 1 - 1e-4),
                    ],
                    -1,
                )
                data_scale = torch.ones_like(data_mean)
                data_scale[..., 0] = 0.1 # adjust these coeffs
                data_scale[..., 1] = 0.01 #

                comp = torch.distributions.Independent(
                    torch.distributions.Laplace(data_mean, data_scale), 1
                )
                gmm = torch.distributions.MixtureSameFamily(mix, comp)
                magnet_list.append(gmm)
                
            for i in range(agent_num):
                l2_norm = (torch.norm(auto_pred['loc_refine_pos'][choose_agent[i],:,:offset, :2] -
                                    sample_action_list[i][1:6], p=2, dim=-1) * reg_mask_list[i][timestep:timestep+offset].unsqueeze(0)).sum(dim=-1)
                action_suggest_index=l2_norm.argmin(dim=-1)
                best_mode[choose_agent[i]] = action_suggest_index

            (
                new_data,
                auto_pred,
                _,
                _,
                (new_true_trans_position_propose, new_true_trans_position_refine),
                (traj_propose, traj_refine),
            ) = get_auto_pred(
                new_data,
                model,
                auto_pred["loc_refine_pos"][
                    torch.arange(traj_propose.size(0)), best_mode
                ],
                auto_pred["loc_refine_head"][
                    torch.arange(traj_propose.size(0)), best_mode, :, 0
                ],
                offset,
                anchor=(init_origin, init_theta, init_rot_mat),
            )

            next_state_temp_list = []
            global_next_state = auto_pred["first_m"]
            for i in range(agent_num):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
            for i in range(agent_num):
                transition_list[batch]["observations"][i].append(state_temp_list[i])
                transition_list[batch]["actions"][i].append(action_list[i])
                transition_list[batch]["next_observations"][i].append(next_state_temp_list[i])
                transition_list[batch]["magnet"][i].append(magnet_list[i])
            if timestep == model.num_future_steps - offset:
                transition_list[batch]["dones"].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]["dones"].append(torch.tensor(0).cuda())
            for i in range(agent_num):
                reward = reward_function(
                    new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map,dataset_type=dataset_type
                )
                transition_list[batch]["rewards"][i].append(
                    torch.tensor([reward]).cuda()
                )
                cost = cost_function(new_data.clone(), model, choose_agent[i], args.distance_limit, choose_agent)
                transition_list[batch]["costs"][i].append(torch.tensor([cost]).cuda())

            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred["pi"], dim=-1)           

def gameformer_baseline_process_batch(
        new_input_data,
        model,
        choose_agent,
        offset,
        scenario_static_map,
        transition_list,
        threeD,
        agent_num
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        pred[f'level_3_interactions'] = pred[f'level_3_interactions'].squeeze(0)
        traj_propose = pred[f'level_3_interactions'][..., :2]

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )

        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            traj_propose[..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred[f'level_3_scores'].squeeze(0)
        pi_eval = F.softmax(pi, dim=-1)
        best_mode = pi_eval.argmax(dim=-1)
        new_true_trans_position_refine = new_true_trans_position_refine[torch.arange(traj_propose.size(0)), best_mode, :, :2]
        new_true_trans_position_refine = torch.cat([new_data['agent']['position'][:,model.num_historical_steps-1,:2].unsqueeze(1),new_true_trans_position_refine], dim=1)
        
        pred_position = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)
        pred_velocity = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)

        for timestep in range(0, model.num_future_steps, offset):
           
            for i in range(agent_num):
                new_data['agent']['position'][:,model.num_historical_steps-offset:model.num_historical_steps,:2] = new_true_trans_position_refine[:,timestep:timestep+offset,:]
                reward = reward_function(
                    new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map)
                transition_list["rewards"][i].append(
                    torch.tensor([reward]).cuda()
                )

        pred_position = new_true_trans_position_refine
        pred_velocity = (
            new_true_trans_position_refine[:,1:,:]-new_true_trans_position_refine[:,:-1,:]
            ) / 0.1
            
        if threeD:
            return pred_position, pred_velocity

def optimal_value_batch(
        args,
        batch,
        new_input_data,
        model,
        agents: List[MAPPO],
        choose_agent,
        offset,
        scenario_static_map,
        scenario_num,
        transition_list,
        render,
        agent_num,
        dataset_type='av2',
        optimal_value=PPO,
        choose_index=0
    ):

        new_data = new_input_data.cuda().clone()

        pred = model(new_data)
        if model.output_head:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["loc_propose_head"],
                    pred["scale_propose_pos"][..., : model.output_dim],
                    pred["conc_propose_head"],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["loc_refine_head"],
                    pred["scale_refine_pos"][..., : model.output_dim],
                    pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            traj_propose = torch.cat(
                [
                    pred["loc_propose_pos"][..., : model.output_dim],
                    pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            traj_refine = torch.cat(
                [
                    pred["loc_refine_pos"][..., : model.output_dim],
                    pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

        auto_pred = pred

        init_origin, init_theta, init_rot_mat, init_v = get_v_transform_mat(
            new_data, model
        )
        transformed_v = torch.einsum("bi,bij->bj", init_v, init_rot_mat)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            init_rot_mat.swapaxes(-1, -2),
        ) + init_origin[:, :2].unsqueeze(1).unsqueeze(1)
        pi = pred["pi"]
        pi_eval = F.softmax(pi, dim=-1)

        state_temp_list = []
        global_state = pred["first_m"]
        for i in range(agent_num):
            state_temp_list.append(global_state[choose_agent[i]])
        frames = []
        pred_position = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)
        pred_heading = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
        pred_velocity = torch.zeros(new_data['agent']['num_nodes'], model.num_future_steps, 2, dtype=torch.float)

        for timestep in range(0, model.num_future_steps, offset):

            best_mode = pi_eval.argmax(dim=-1)

            magnet_list = []
            action_list = []
            true_trans_position_refine=new_true_trans_position_refine

            for i in range(agent_num):
                pred_v = (
                    pred["loc_refine_pos"][choose_agent[i], :, 1 : offset + 2]
                    - pred["loc_refine_pos"][choose_agent[i], :, : offset + 1]
                ) / 0.1
                a = (
                    pred_v[:, 1 : offset + 1].norm(dim=-1)
                    - pred_v[:, :offset].norm(dim=-1)
                ) / 0.1
                kappa = (
                    pred["loc_refine_head"][choose_agent[i], :, 1 : offset + 1, 0]
                    - pred["loc_refine_head"][choose_agent[i], :, :offset, 0]
                ) / (pred_v[:, :offset].norm(dim=-1) * 0.1 + 1 / 2 * (0.1**2) * a)
                a = a.mean(-1)
                kappa = kappa.mean(-1)
                if i==choose_index:
                    action = optimal_value.choose_action(state_temp_list[i].flatten()[None, :])[
                        0
                    ]
                else:
                    action = agents[i].choose_action(state_temp_list[i].flatten()[None, :])[
                        0
                    ]
                action_a = action[0].clip(-1,1) * 5
                action_kappa = action[1].clip(-1,1) * 0.05

                v = transformed_v[choose_agent[i]]
                newpos = torch.zeros(offset + 2, 2)
                newhead = torch.zeros(offset + 2)
                for t in range(offset + 1):
                    newpos[t + 1, 0] = (
                        newpos[t, 0]
                        + v[0] * 0.1
                        + 1 / 2 * action_a * torch.cos(newhead[t]) * (0.1) ** 2
                    )
                    newpos[t + 1, 1] = (
                        newpos[t, 1]
                        + v[1] * 0.1
                        + 1 / 2 * action_a * torch.sin(newhead[t]) * (0.1) ** 2
                    )
                    newhead[t + 1] = newhead[t] + action_kappa * (
                        v.norm() * 0.1 + 1 / 2 * action_a * (0.1) ** 2
                    )
                    v_abs = v.norm() + action_a * 0.1
                    v[0] = v_abs * torch.cos(newhead[t + 1])
                    v[1] = v_abs * torch.sin(newhead[t + 1])

                auto_pred["loc_refine_pos"][
                    choose_agent[i],
                    best_mode[choose_agent[i]],
                    : offset + 1,
                    : model.output_dim,
                ] = newpos[1:]
                auto_pred["loc_refine_head"][
                    choose_agent[i], best_mode[choose_agent[i]], : offset + 1
                ] = newhead[1:].unsqueeze(-1)

                action_list.append(action)
                mix = torch.distributions.Categorical(pi_eval[choose_agent[i]])
                data_mean = torch.stack(
                    [
                        (a / 5).clip(-1 + 1e-4, 1 - 1e-4),
                        (kappa / 0.05).clip(-1 + 1e-4, 1 - 1e-4),
                    ],
                    -1,
                )
                data_scale = torch.ones_like(data_mean)
                data_scale[..., 0] = 0.1 # adjust these coeffs
                data_scale[..., 1] = 0.01 #

                comp = torch.distributions.Independent(
                    torch.distributions.Laplace(data_mean, data_scale), 1
                )
                gmm = torch.distributions.MixtureSameFamily(mix, comp)
                magnet_list.append(gmm)

            (
                new_data,
                auto_pred,
                _,
                _,
                (new_true_trans_position_propose, new_true_trans_position_refine),
                (traj_propose, traj_refine),
            ) = get_auto_pred(
                new_data,
                model,
                auto_pred["loc_refine_pos"][
                    torch.arange(traj_propose.size(0)), best_mode
                ],
                auto_pred["loc_refine_head"][
                    torch.arange(traj_propose.size(0)), best_mode, :, 0
                ],
                offset,
                anchor=(init_origin, init_theta, init_rot_mat),
            )

            next_state_temp_list = []
            global_next_state = auto_pred["first_m"]
            for i in range(agent_num):
                next_state_temp_list.append(global_next_state[choose_agent[i]])
            for i in range(agent_num):
                transition_list[batch]["observations"][i].append(state_temp_list[i])
                transition_list[batch]["actions"][i].append(action_list[i])
                transition_list[batch]["next_observations"][i].append(next_state_temp_list[i])
                transition_list[batch]["magnet"][i].append(magnet_list[i])
            if timestep == model.num_future_steps - offset:
                transition_list[batch]["dones"].append(torch.tensor(1).cuda())
            else:
                transition_list[batch]["dones"].append(torch.tensor(0).cuda())
            for i in range(agent_num):
                reward = reward_function(
                    new_input_data.clone(), new_data.clone(), model, choose_agent[i], scenario_static_map, dataset_type=dataset_type
                )
                transition_list[batch]["rewards"][i].append(
                    torch.tensor([reward]).cuda()
                )
                cost = cost_function(new_data.clone(), model, choose_agent[i], args.distance_limit, choose_agent)
                transition_list[batch]["costs"][i].append(torch.tensor([cost]).cuda())

            state_temp_list = next_state_temp_list
            pi_eval = F.softmax(auto_pred["pi"], dim=-1)
            if render:
                for t in range(offset):
                    plot_traj_with_data(
                        new_data,
                        scenario_static_map,
                        agent_number=agent_num,
                        scenario_num=scenario_num,
                        bounds=50,
                        t=11 - offset + t,
                        dataset_type=dataset_type,
                        choose_agent=choose_agent
                    )
                    # for agent in range(agent_num):
                    #     for j in range(6):
                    #         xy = true_trans_position_refine[choose_agent[agent]].cpu()
                    #         plt.plot(xy[j, ..., 0], xy[j, ..., 1])
                    # plot_destination(args.scenario)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    frame = img.open(buf)
                    frames.append(frame)

                pred_position[:,timestep:timestep+offset,:] = new_data['agent']['position'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
                pred_heading[:,timestep:timestep+offset] = new_data['agent']['heading'][:,model.num_historical_steps-offset:model.num_historical_steps]
                pred_velocity[:,timestep:timestep+offset,:] = new_data['agent']['velocity'][:,model.num_historical_steps-offset:model.num_historical_steps,:2]
            
        if render:
            import imageio

            # Specify the file path for the GIF
            gif_path = generate_tmp_gif_path()
            tmp = imageio.mimsave(gif_path, frames, fps=10)
            # run["val/rollout"].append(File(gif_path))
            if args.track:
                wandb.log({"val/rollout": wandb.Video(gif_path, format="gif")})
        
            return pred_position, pred_heading, pred_velocity
        