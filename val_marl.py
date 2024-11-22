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
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors.autoval import AutoQCNet
from transforms import TargetBuilder
from utils.rollout import PPO_process_batch, expert_process_batch
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
from algorithm.TrafficGamer import TrafficGamer
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
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument(
        "--ckpt_path", default="./checkpoints/epoch=19-step=499780.ckpt", type=str
    )
    parser.add_argument("--RL_config", default="TrafficGamer_eval", type=str)
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

    parser.add_argument("--eta-coef1", type=float, default=0.05)
    parser.add_argument("--eta-coef2", type=float, default=0.1)
    parser.add_argument("--workspace", type=str, default='TrafficGamer')
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
    
    offset = config["offset"]
    config['eta_coef1'] = args.eta_coef1
    config['eta_coef2'] = args.eta_coef2
    config['is_magnet'] = args.magnet
    config['agent_number'] = agent_num
    config['penalty_initial_value'] = args.penalty_initial_value

    agents = [TrafficGamer(
            state_dim=model.num_modes * config["hidden_dim"],
            agent_number=agent_num,
            config=config,
            device=model.device,
        ) for _ in range(agent_num)]
    
    model_state_dict = torch.load('./save_models/scenario3/scenario3_distance_limit8.0_penalty_initial_value1.0_cost_quantile56_waymo_2024_08_15_23_34_24.pth')
    for i, agent in enumerate(agents):
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
                "states": [[] for _ in range(agent_num)],
                "actions": [[] for _ in range(agent_num)],
                "next_states": [[] for _ in range(agent_num)],
                "rewards": [[] for _ in range(agent_num)],
                "magnet": [[] for _ in range(agent_num)],
                "costs": [[] for _ in range(agent_num)],
                "dones": [],
            }
            for _ in range(config["batch_size"])
        ]
        with torch.no_grad():
            pred_position, pred_heading, pred_velocity = PPO_process_batch(
                args,
                0,
                new_input_data,
                model,
                agents,
                choose_agent,
                offset,
                scenario_static_map,
                args.scenario,
                transition_list,
                render=True,
                agent_num=agent_num
            )
            
            pred_p = pred_position
            pred_h = pred_heading
            pred_v = pred_velocity
            
    current_time = time.time()
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(current_time))
        
    if args.threeD:
        pred_p = torch.cat([new_input_data['agent']['position'][:,:model.num_historical_steps,:2], pred_p.cuda()], dim=-2)
        pred_h = torch.cat([new_input_data['agent']['heading'][:,:model.num_historical_steps], pred_h.cuda()], dim=-1)
        pred_v = torch.cat([new_input_data['agent']['velocity'][:,:model.num_historical_steps,:2], pred_v.cuda()], dim=-2)

        df = pd.read_parquet(os.path.join(argoverse_scenario_dir, scenario_id, f'scenario_{scenario_id}.parquet'))
        length = len(df)
        df2 = df
        historical_df = df[df['timestep'] < model.num_historical_steps]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]
        agent_ids = list(df['track_id'].unique())
        for track_id, track_df in df2.groupby('track_id'):
            if track_id in agent_ids:
                agent_idx = agent_ids.index(track_id)
                agent_steps = track_df['timestep'].values
                for t in range(len(track_df)):
                    df2.loc[track_df.index[t], 'position_x'] = pred_p[agent_idx, t, 0].cpu().numpy()
                    df2.loc[track_df.index[t], 'position_y'] = pred_p[agent_idx, t, 1].cpu().numpy()
                    df2.loc[track_df.index[t], 'heading'] = pred_h[agent_idx, t].cpu().numpy()
                    df2.loc[track_df.index[t], 'velocity_x'] = pred_v[agent_idx, t, 0].cpu().numpy()
                    df2.loc[track_df.index[t], 'velocity_y'] = pred_v[agent_idx, t, 1].cpu().numpy()

        for i in range(agent_num-1):
            for j in range(model.num_historical_steps + model.num_future_steps):
                l = length+i*(model.num_historical_steps + model.num_future_steps)+j
                df2.loc[l, 'observed'] = True
                df2.loc[l, 'track_id'] = new_input_data['agent']['id'][0][data['agent']['num_nodes'] + i]
                df2.loc[l, 'object_type'] = 'vehicle'
                df2.loc[l, 'object_category'] = 2
                df2.loc[l, 'timestep'] = int(j)
                df2.loc[l, 'position_x'] = pred_p[data['agent']['num_nodes'] + i, j, 0].cpu().numpy()
                df2.loc[l, 'position_y'] = pred_p[data['agent']['num_nodes'] + i, j, 1].cpu().numpy()
                df2.loc[l, 'heading'] = pred_h[data['agent']['num_nodes'] + i, j].cpu().numpy()
                df2.loc[l, 'velocity_x'] = pred_v[data['agent']['num_nodes'] + i, j, 0].cpu().numpy()
                df2.loc[l, 'velocity_y'] = pred_v[data['agent']['num_nodes'] + i, j, 1].cpu().numpy()
                df2.loc[l, 'scenario_id'] = scenario_id
                df2.loc[l, 'start_timestamp'] = int(df['start_timestamp'][0])
                df2.loc[l, 'end_timestamp'] = int(df['end_timestamp'][0])
                df2.loc[l, 'num_timestamps'] = int(df['num_timestamps'][0])
                df2.loc[l, 'focal_track_id'] = new_input_data['agent']['id'][0][agent_index]
                df2.loc[l, 'city'] = df['city'][0]

        destination_folder = f'./scenario{args.scenario}/'+str(scenario_id)+f'_distance_limit{args.distance_limit}_penalty_initial_value{args.penalty_initial_value}_cost_quantile{args.cost_quantile}_seed{args.seed}'
        destination_folder = os.path.expanduser(destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        destination_path = os.path.join(destination_folder, os.path.basename(static_map_path))
        shutil.copy(static_map_path, destination_path)
        df2.to_parquet(
            Path(destination_folder) / f'scenario_{scenario_id}_{time_string}.parquet'
        )



        
