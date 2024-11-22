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
import neptune
import wandb
import pytorch_lightning as pl
import torch, math
import yaml, time
import numpy as np
import torch.nn.functional as F
import pandas as pd
from algorithm.mappo import MAPPO
from algorithm.cce_mappo import CCE_MAPPO
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets.waymo_dataset import WaymoDataset
from predictors.autoval import AutoQCNet
from transforms import TargetBuilder
# from utils.rollout import PPO_process_batch, expert_process_batch
from utils.rollout import PPO_process_batch, PPO_process_batch_confined_actions
from utils.utils import seed_everything
from utils.data_utils import expand_data
from itertools import chain, compress
from neptune.types import File
from torch_geometric.data import Batch
from tqdm import tqdm
from distutils.util import strtobool
from metadrive.scenario.utils import draw_map
import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace
import numpy as np
from tqdm import tqdm
from pathlib import Path
from algorithm.TrafficGamer import TrafficGamer
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
import json
try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object
from metadrive.type import MetaDriveType
from matplotlib.figure import Figure
import pickle
import numpy as np
import math
from shapely.geometry import Point, Polygon, LineString
# def draw_map(map_features, show=False, save=False, save_path="map.png"):
#     fig: Figure = plt.figure(figsize=(8, 6), dpi=500)
#     ax = fig.add_subplot(111)
    
#     type_attribute = {}
#     # breakpoint()
#     for key, value in map_features.items():
#         print(key, value['type'], value.keys())
#         type_attribute[value['type']] = list(value.keys())
#         if MetaDriveType.is_lane(value.get("type", None)):
#             ax.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1)
#         elif value.get("type", None) == "road_edge":
#             ax.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1, c=(0, 0, 0))
#         # 如果有其他类型的特征需要绘制，可以继续添加
#     print(type_attribute)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal')
    
#     if save:
#         fig.savefig(save_path, bbox_inches='tight')
    
#     if show:
#         plt.show()
    
#     plt.close(fig)
def continuous_valid_length(valid_mask):
    max_length = 0
    current_length = 0
    
    for is_valid in valid_mask:
        if is_valid:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
            current_length = 0
    
    # 检查最后一段连续True的长度
    if current_length > max_length:
        max_length = current_length
    
    return max_length
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="QCNet")
    parser.add_argument(
        "--root",
        type=str,
        default="~/Multi-agent-competitive-environment/datasets",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--scenario", type=int, default=1)

    parser.add_argument(
        "--ckpt_path", default="./Multi-agent-competitive-environment/epoch=20-step=79905.ckpt", type=str
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
        default=False,
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
    parser.add_argument("--eta-coef2", type=float, default=0.05)
    parser.add_argument("--workspace", type=str, default='TrafficGamer')
    parser.add_argument("--distance_limit", type=float, default=5.0)
    parser.add_argument("--penalty_initial_value", type=float, default=1.0)
    parser.add_argument("--cost_quantile", type=int, default=48)
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
        'waymo': WaymoDataset,
    }[model.dataset](
        root=args.root,
        processed_dir="./Multi-agent-competitive-environment/datasets/validation/processed",
        split="val",
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
        is_select=True
    )

    scene_id = [
        '9c6eb32bcc69d42e',
        '2f1be7eedc2c7333',
        'ab68832bf7312ab3',
        '63bcffc229444c56',
        'caea26e357c20bfc',
        '5d0c97b991689cde'
    ]
    agent_id_list = [
        [0,1,2,3,137,42,52],
        [983,994,987,980,1007,1475],
        [387,354,352,384,385,386],
        [230,758,713,714,715],
        [2159,2160,2161,2162,2163,2164],
        [885,1188,887,888,890],
    ]

    scenarionet_pkl = f'sd_waymo_v1.2_{scene_id[args.scenario-1]}.pkl'
    dataloader = DataLoader(
        val_dataset[[val_dataset.processed_file_names.index(f'{scene_id[args.scenario-1]}.pkl')]],
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
    
    df_mapping = pd.read_pickle('./scenarionet/datasets/way_convert/dataset_mapping.pkl')
    pkl_loc = df_mapping[scenarionet_pkl]
    df_summary = pd.read_pickle(f'./scenarionet/datasets/way_convert/{pkl_loc}/dataset_summary.pkl')
    df = pd.read_pickle(f'./scenarionet/datasets/way_convert/{pkl_loc}/{scenarionet_pkl}')
    scenario_static_map = df['map_features']

    # draw_map(df['map_features'], save=True)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
    agent_index = torch.nonzero(data["agent"]["target_mask"], as_tuple=False)[-1].item()
    
    agent_num=len(agent_id_list[args.scenario-1])
    agent_rewards = [[] for _ in range(agent_num)]
    expert_rewards = 0
    offset = config["offset"]
    config['eta_coef1'] = args.eta_coef1
    config['eta_coef2'] = args.eta_coef2
    config['is_magnet'] = args.magnet
    config['agent_number'] = agent_num
    config['penalty_initial_value'] = args.penalty_initial_value
    
    choose_agent = [i for i, j in enumerate(data['agent']['id']) if j in agent_id_list[args.scenario-1]]
    print(choose_agent)

    agents = [TrafficGamer(
            state_dim=model.num_modes * config["hidden_dim"],
            agent_number=agent_num,
            config=config,
            device=model.device,
        ) for _ in range(agent_num)]
    
    model_state_dict = torch.load('./save_models/scenario1/scenario1_distance_limit5.0_penalty_initial_value1.0_cost_quantile8_waymo_2024_08_13_04_45_36.pth')
    for i, agent in enumerate(agents):
            agent.pi.load_state_dict(model_state_dict[f'agent_{i}_pi'])
            agent.pi.eval()
    
    pred_p = torch.zeros(data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)
    pred_h = torch.zeros(data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
    pred_v = torch.zeros(data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)
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
            pred_position, pred_heading, pred_velocity = PPO_process_batch(
                args,
                0,
                data,
                model,
                agents,
                choose_agent,
                offset,
                scenario_static_map,
                args.scenario,
                transition_list,
                render=True,
                agent_num=agent_num,
                dataset_type='waymo'
            )
            
            pred_p = pred_position
            pred_h = pred_heading
            pred_v = pred_velocity
        
    current_time = time.time()
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(current_time))
    
    if args.threeD:
        pred_p = torch.cat([data['agent']['position'][:,:model.num_historical_steps,:2], pred_p.cuda()], dim=-2)
        pred_h = torch.cat([data['agent']['heading'][:,:model.num_historical_steps], pred_h.cuda()], dim=-1)
        pred_v = torch.cat([data['agent']['velocity'][:,:model.num_historical_steps,:2], pred_v.cuda()], dim=-2)
        agent_ids = [str(i) for i in agent_id_list[args.scenario-1]]
        df['metadata']['objects_of_interest'] = agent_ids
        diff = set(agent_ids) - set(df['tracks'].keys())
        df['metadata']['number_summary']['num_objects'] += len(diff)
        df['metadata']['number_summary']['num_objects_each_type']['VEHICLE'] += len(diff)
        df['metadata']['number_summary']['num_moving_objects']+= len(diff)
        df['metadata']['number_summary']['num_moving_objects_each_type']['VEHICLE']+= len(diff)
        for i in range(agent_num):
            track_id = str(data['agent']['id'][choose_agent[i]].item())
            #remember to change scenarionet pkl file
            if track_id in list(df['tracks'].keys()) and df['tracks'][track_id]['type']=='VEHICLE':
                pos = df['tracks'][track_id]['state']['position']
                heading = df['tracks'][track_id]['state']['heading']
                vel = df['tracks'][track_id]['state']['velocity']
                valid = df['tracks'][track_id]['state']['valid']
                total_distance = 0
                pos_shape = pos.shape
                if valid[:11].sum()!=11:
                    valid = np.zeros(91,)
                else:
                    valid = np.ones(91,)
                for j in range(model.num_historical_steps + model.num_future_steps):

                    pos[j,0] = pred_p[choose_agent[i], j, 0].cpu().numpy()
                    pos[j,1] = pred_p[choose_agent[i], j, 1].cpu().numpy()
                    heading[j] = pred_h[choose_agent[i], j].cpu().numpy()
                    vel[j,0]=pred_v[choose_agent[i], j, 0].cpu().numpy()
                    vel[j,1]=pred_v[choose_agent[i], j, 1].cpu().numpy()
                    # valid[i]=state_cor['observed'].iloc[0]
                    if (i>=1) and (np.any(pos[j-1,:2]!=0)):
                        x1, y1 = pos[j - 1,:2]
                        x2, y2 = pos[j,:2]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        total_distance += distance
                    # print(valid.sum())
                    df['metadata']['object_summary'][track_id]={'type':'VEHICLE','object_id':track_id,'track_length':91, 'moving_distance':total_distance, 'valid_length':valid.sum(), 'continuous_valid_length':continuous_valid_length(valid)}
                    df['tracks'][track_id]['state']['length'] *= valid
                    df['tracks'][track_id]['state']['width'] *= valid
                    df['tracks'][track_id]['state']['height'] *= valid
                    df['tracks'][track_id]['state']['position'] = pos
                    df['tracks'][track_id]['state']['heading'] = heading
                    df['tracks'][track_id]['state']['velocity'] = vel
                    df['tracks'][track_id]['state']['valid'] = valid
            else:
                n = model.num_historical_steps + model.num_future_steps
                pos = np.zeros((n,3))
                heading = np.zeros((n,))
                vel = np.zeros((n,2))
                length = np.zeros((n,))
                width = np.zeros((n,))
                height = np.zeros((n,))
                valid = np.ones((91,))
                length = 4.0*valid
                width=1.6*valid
                height = 1.0*valid
                total_distance = 0.0
                for j in range(n):
                    pos[j,0] = pred_p[choose_agent[i], j, 0].cpu().numpy()
                    pos[j,1] = pred_p[choose_agent[i], j, 1].cpu().numpy()
                    heading[j] = pred_h[choose_agent[i], j].cpu().numpy()
                    vel[j,0]=pred_v[choose_agent[i], j, 0].cpu().numpy()
                    vel[j,1]=pred_v[choose_agent[i], j, 1].cpu().numpy()
                    if (j>=1) and (np.any(pos[j-1,:2]!=0)):
                        x1, y1 = pos[j - 1,:2]
                        x2, y2 = pos[j,:2]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        total_distance += distance
                df['metadata']['object_summary'][track_id]={'type':'VEHICLE','object_id':track_id,'track_length':91, 'moving_distance':total_distance, 'valid_length':valid.sum(), 'continuous_valid_length':continuous_valid_length(valid)}
                tracklet = {}
                tracklet['type'] = 'VEHICLE'
                tracklet['metadata'] = {'track_length': 91, 'type': 'VEHICLE', 'object_id': track_id, 'dataset': 'waymo'}
                tracklet['state'] = {'position':pos, 'length':length, 'width':width, 'height':height, 'heading':heading, 'velocity':vel, 'valid':valid}
                df['tracks'][track_id]=tracklet
        current_time = time.time()
        time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(current_time))
        filename = f'./waymo_pred/scenario{args.scenario}/sd_waymo_v1.2_{scene_id[args.scenario-1]}_scenario{args.scenario}_distance_limit{args.distance_limit}_penalty_initial_value{args.penalty_initial_value}_cost_quantile{args.cost_quantile}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(df, f)