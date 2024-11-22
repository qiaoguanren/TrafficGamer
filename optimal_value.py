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
# import neptune
import wandb
import pytorch_lightning as pl
import torch, math
import yaml, time
import numpy as np
import torch.nn.functional as F
import pandas as pd
from algorithm.mappo import MAPPO
from algorithm.ppo import PPO
from algorithm.cce_mappo import CCE_MAPPO
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors.autoval import AutoQCNet
from transforms import TargetBuilder
from utils.rollout import PPO_process_batch, expert_process_batch,optimal_value_batch
from utils.utils import seed_everything
from utils.data_utils import expand_data
from itertools import chain, compress
from neptune.types import File
from torch_geometric.data import Batch
from tqdm import tqdm
from distutils.util import strtobool
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os, shutil
from algorithm.TraffcGamer import TraffcGamer
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="QCNet")
    parser.add_argument(
        "--root",
        type=str,
        default="./Multi-agent-competitive-environment/datasets",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--scenario", type=int, default=5)
    parser.add_argument(
        "--ckpt_path", default="./checkpoints/epoch=19-step=499780.ckpt", type=str
    )
    parser.add_argument("--RL_config", default="MAPPO_seed666", type=str)
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked",
    )
    parser.add_argument(
        "--magnet",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--threeD",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, the output data will be stored",
    )

    parser.add_argument("--eta-coef1", type=float, default=0.02)
    parser.add_argument("--eta-coef2", type=float, default=0.1)
    parser.add_argument("--workspace", type=str, default='Constrainted-TrafficGamer')
    parser.add_argument("--distance_limit", type=float, default=2.0)
    parser.add_argument("--penalty_initial_value", type=float, default=1.0)
    parser.add_argument("--cost_quantile", type=int, default=8)
    
    args = parser.parse_args()
    if not args.magnet:
        args.eta_coef1 = 0.0
    with open("configs/" + args.RL_config + ".yaml", "r") as file:
        config = yaml.safe_load(file)
    file.close()
    pl.seed_everything(args.seed, workers=True)
    seed_everything(args.seed)

    model = {
        "QCNet": AutoQCNet,
    }[
        args.model
    ].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    val_dataset = {
        "argoverse_v2": ArgoverseV2Dataset,
    }[model.dataset](
        root=args.root,
        split="val",
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
    )

    scene_id = ['d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84',
                '00a50e9f-63a1-4678-a4fe-c6109721ecba',
                '236df665-eec6-4c25-8822-950a6150eade',
                'cb0133ff-f7ad-43b7-b260-7068ace15307',
                'cdf70cc8-d13d-470b-bb39-4f1812acc146',
                '3856ed37-4a05-4131-9b12-c4f4716fec92']

    dataloader = DataLoader(
        val_dataset[[val_dataset.processed_file_names.index(scene_id[args.scenario-1]+'.pkl')]],
        #val_dataset[[val_dataset.raw_file_names.index(scene_id[args.scenario-1])]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    if args.track:
        # run = neptune.init_run(
        #     project="mcgubio/"+args.workspace,
        #     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNWJjMjk1Zi01ODBhLTRhZjctOTM0Ni0zMjU0NGM5ZTgwMWYifQ=='
        # )
        # run["parameters"] = vars(args)
        run = wandb.init(
            project=args.workspace,
            config={
                "penalty_initial_value": args.penalty_initial_value,
                "distance_limit": args.distance_limit,
                "eta-coef1": args.eta_coef1,
                "eta-coef2": args.eta_coef2,
                "cost_quantile": args.cost_quantile,
            },
            entity="greywhale",
            name="scenario-" + str(args.scenario)+"-piv_"+str(args.penalty_initial_value)+"-dl_"+str(args.distance_limit)+"-seed_"+str(args.seed),
        )
    else:
        run = {}

    it = iter(dataloader)
    data = next(it)
    scenario_id = data["scenario_id"][0]
    argoverse_scenario_dir = Path(
        "~/Multi-agent-competitive-environment/datasets/val/raw"
    )

    all_scenario_files = sorted(
        argoverse_scenario_dir.rglob(f"*_{scenario_id}.parquet")
    )
    scenario_file_list = list(all_scenario_files)
    scenario_path = scenario_file_list[0]

    static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    scenario_static_map = ArgoverseStaticMap.from_json(static_map_path)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

    agent_index = torch.nonzero(data["agent"]["category"] == 3, as_tuple=False).item()
    new_input_data = expand_data(data, args.scenario, agent_index)
    
    agent_num=new_input_data["agent"]["num_nodes"]-data["agent"]["num_nodes"]+1
    agent_rewards = [[] for _ in range(agent_num)]
    expert_rewards = 0
    choose_index = 1
    
    offset = config["offset"]
    config['eta_coef1'] = args.eta_coef1
    config['eta_coef2'] = args.eta_coef2
    config['is_magnet'] = args.magnet
    config['agent_number'] = agent_num
    config['penalty_initial_value'] = args.penalty_initial_value

    agents = [CCE_MAPPO(
            state_dim=model.num_modes * config["hidden_dim"],
            agent_number=agent_num,
            config=config,
            device=model.device,
        ) for _ in range(agent_num)]

    optimal_value = PPO(
            state_dim=model.num_modes * config["hidden_dim"],
            agent_number=1,
            config=config,
            device=model.device,
        )
    
    model_state_dict = torch.load('./save_models/scenario6/scenario6_distance_limit2.0_penalty_initial_value5.0_cost_quantile8_seed123_2024_07_15_22_08_00.pth')
    for i, agent in enumerate(agents):
            if i!=choose_index:
                agent.pi.load_state_dict(model_state_dict[f'agent_{i}_pi'])
                agent.pi.eval()
            
    choose_agent = []
    choose_agent.append(agent_index)
    for i in range(agent_num - 1):
        choose_agent.append(data["agent"]["num_nodes"] + i)
        
    pred_p = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)
    pred_h = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
    pred_v = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)

    for episode in tqdm(range(config["episodes"])):
        transition_list = [
            {
                "observations": [[] for _ in range(agent_num)],
                "actions": [[] for _ in range(agent_num)],
                "next_observations": [[] for _ in range(agent_num)],
                "rewards": [[] for _ in range(agent_num)],
                "magnet": [[] for _ in range(agent_num)],
                "costs": [[] for _ in range(agent_num)],
                "dones": [],
            }
            for _ in range(config["batch_size"])
        ]
        with torch.no_grad():
            for batch in range(config["batch_size"]):
                
                # pred_p = pred_position
                # pred_h = pred_heading
                # pred_v = pred_velocity
                optimal_value_batch(
                    args,
                    batch,
                    new_input_data,
                    model,
                    agents,
                    choose_agent,
                    offset,
                    scenario_static_map,
                    args.scenario,
                    transition_list,
                    render=False,
                    agent_num=agent_num,
                    dataset_type='av2',
                    optimal_value=optimal_value,
                    choose_index=choose_index,
                )

        # for i in range(agent_num):
        logs = optimal_value.update(transition_list,choose_index)
        if args.track:
            for key in logs[0].keys():
                if args.track:
                    wandb.log({f"agent_{choose_index}_{key}": np.mean([d[key] for d in logs], axis=0)}, step=episode)
                # run[f"agent_{i}_{key}"].append(np.mean([d[key] for d in logs], axis=0),step=episode)

        undiscounted_return = 0
        for t in reversed(range(0, int(model.num_future_steps / offset))):
            _sum = 0
            
            for b in range(config["batch_size"]):
                _sum += float(transition_list[b]["rewards"][choose_index][t])
            mean_reward = _sum / config["batch_size"] # if expert reward<agent reward, we can only consider undiscounted_return += max_reward, where max_reward is the max reward of batchs
            undiscounted_return += mean_reward
        # agent_rewards[i].append(undiscounted_return)
        if args.track:
            wandb.log({f"agent_{choose_index}_return": undiscounted_return},step=episode)
            # run[f"agent_{i}_return"].append(undiscounted_return)

wandb.finish()



        
