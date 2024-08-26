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
import os, shutil
import pytorch_lightning as pl
import torch, math
import yaml, time
import pandas as pd
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from datasets import ArgoverseV2Dataset
from predictors.autoval import AntoQCNet
from transforms import TargetBuilder
from utils.rollout import qcnet_baseline_process_batch
from utils.utils import seed_everything
from utils.data_utils import expand_data
from torch_geometric.data import Batch
from tqdm import tqdm
from distutils.util import strtobool
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm
from pathlib import Path

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
        default="~/Multi-agent-competitive-environment/datasets",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument(
        "--ckpt_path", default="~/Multi-agent-competitive-environment/checkpoints/epoch=19-step=499780.ckpt", type=str
    )
    parser.add_argument("--RL_config", default="TrafficGamer", type=str)
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
    parser.add_argument("--distance_limit", type=float, default=5.0)
    parser.add_argument("--penalty_initial_value", type=float, default=1.0)
    parser.add_argument("--cost_quantile", type=int, default=48)
    
    args = parser.parse_args()
    with open("configs/" + args.RL_config + ".yaml", "r") as file:
        config = yaml.safe_load(file)
    file.close()
    pl.seed_everything(args.seed, workers=True)
    seed_everything(args.seed)

    model = {
        "QCNet": AntoQCNet,
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
    
    offset = config["offset"]
            
    choose_agent = []
    choose_agent.append(agent_index)
    for i in range(agent_num - 1):
        choose_agent.append(data["agent"]["num_nodes"] + i)

    pred_p = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)
    pred_h = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, dtype=torch.float)
    pred_v = torch.zeros(new_input_data['agent']['num_nodes'], model.num_future_steps, model.output_dim, dtype=torch.float)

    baseline_rewards = [[] for _ in range(agent_num)]
    df = pd.DataFrame()
    for episode in tqdm(range(config["episodes"])):
        transition_list = {
                "states": [[] for _ in range(agent_num)],
                "next_states": [[] for _ in range(agent_num)],
                "rewards": [[] for _ in range(agent_num)]
            }
        with torch.no_grad():
            if episode == config['episodes'] - 1:
                pred_position, pred_heading, pred_velocity = qcnet_baseline_process_batch(
                    new_input_data,
                    model,
                    choose_agent,
                    offset,
                    scenario_static_map,
                    transition_list,
                    threeD=True,
                    agent_num=agent_num
                )
                pred_p = pred_position
                pred_h = pred_heading
                pred_v = pred_velocity
            else:
                qcnet_baseline_process_batch(
                    new_input_data,
                    model,
                    choose_agent,
                    offset,
                    scenario_static_map,
                    transition_list,
                    threeD=False,
                    agent_num=agent_num
                )

        undiscounted_return = 0
        for i in range(agent_num):
            for t in reversed(range(0, model.num_future_steps // offset)):
                    mean_reward = float(transition_list["rewards"][i][t])
                    undiscounted_return += mean_reward  
            baseline_rewards[i].append(undiscounted_return)
    for i in range(agent_num):
        df[f'agent_{i}'] = baseline_rewards[i]

    #df.to_csv(f'./cce-gap/qcnet_baseline_rewards_scenario{args.scenario}_seed{args.seed}.csv', index=False)
    
    if args.threeD:
        pred_p = torch.cat([new_input_data['agent']['position'][:,:model.num_historical_steps,:2], pred_p.cuda()], dim=-2)
        pred_h = torch.cat([new_input_data['agent']['heading'][:,:model.num_historical_steps], pred_h.cuda()], dim=-1)
        pred_v = torch.cat([new_input_data['agent']['velocity'][:,:model.num_historical_steps,:2], pred_v.cuda()], dim=-2)

        df = pd.read_parquet(os.path.join(argoverse_scenario_dir, scenario_id, f'scenario_{scenario_id}.parquet'))
        length = len(df)
        df2 = df.copy()
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

        destination_folder = f'./argoverse2_pred/scenario{args.scenario}/'+str(scenario_id)+'_qcnet_baseline'
        destination_folder = os.path.expanduser(destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        destination_path = os.path.join(destination_folder, os.path.basename(static_map_path))
        shutil.copy(static_map_path, destination_path)
        df2.to_parquet(
            Path(destination_folder) / f'scenario_{scenario_id}.parquet'
        )


        
