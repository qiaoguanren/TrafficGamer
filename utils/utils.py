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

import torch, copy, math, os, random
import numpy as np
import tempfile
from utils.geometry import wrap_angle
from shapely.geometry import Point, Polygon, LineString
import neptune
import pytorch_lightning as pl
import torch, math
import yaml
import numpy as np
import torch.nn.functional as F
# from algorithm.mappo import MAPPO
from shapely.geometry import Point, Polygon
import torch
import math, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
from torch.distributions import Normal

def generate_tmp_gif_path():
    # Create a temporary file with a random name and .gif extension
    temp_gif = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)

    # Get the path of the temporary GIF file
    gif_path = temp_gif.name

    # Close the temporary file (we only need its path)
    temp_gif.close()

    return gif_path

def dict_mean(dict_list, agent):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[f"agent_{agent}_{key}"] = np.mean(
                [d[key] for d in dict_list], axis=0
            )
        return mean_dict

def get_v_transform_mat(input_data, model):
    origin = input_data["agent"]["position"][:, model.num_historical_steps - 1]
    v = (
        input_data["agent"]["position"][:, model.num_historical_steps - 1, :2]
        - input_data["agent"]["position"][:, model.num_historical_steps - 2, :2]
    ) / 0.1
    theta = input_data["agent"]["heading"][:, model.num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = theta.new_zeros(input_data["agent"]["num_nodes"], 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    return origin, theta, rot_mat, v

def get_transform_mat(input_data, model):
    origin = input_data["agent"]["position"][:, model.num_historical_steps - 1]
    theta = input_data["agent"]["heading"][:, model.num_historical_steps - 1]
    cos, sin = theta.cos(), theta.sin()
    rot_mat = theta.new_zeros(input_data["agent"]["num_nodes"], 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos
    return origin, theta, rot_mat


def get_auto_pred(
    input_data, model, loc_refine_pos, loc_refine_head, offset, anchor=None
):
    old_anchor = origin, theta, rot_mat = get_transform_mat(input_data, model)
    # auto_index = data['agent']['valid_mask'][:,model.num_historical_steps]
    input_data["agent"]["valid_mask"] = (
        torch.cat(
            (
                input_data["agent"]["valid_mask"][..., offset:],
                torch.zeros(input_data["agent"]["valid_mask"].shape[:-1] + (5,)).cuda(),
            ),
            dim=-1,
        )
    ).bool()
    input_data["agent"]["valid_mask"][:, 0] = False
    new_position = torch.matmul(
        loc_refine_pos[..., :2], rot_mat.swapaxes(-1, -2)
    ) + origin[:, :2].unsqueeze(1)
    input_position = torch.zeros_like(input_data["agent"]["position"])
    input_position[:, :-offset] = input_data["agent"]["position"][:, offset:]
    input_position[
        :, model.num_historical_steps - offset : model.num_historical_steps, :2
    ] = new_position[:, :offset]

    input_v = torch.zeros_like(input_data["agent"]["velocity"])
    input_v[:, :-offset] = input_data["agent"]["velocity"][:, offset:]
    input_v[:, model.num_historical_steps - offset : model.num_historical_steps, :2] = (
        new_position[:, 1:] - new_position[:, :-1]
    )[:, :offset] / 0.1

    input_heading = torch.zeros_like(input_data["agent"]["heading"])
    input_heading[:, :-offset] = input_data["agent"]["heading"][:, offset:]
    input_heading[
        :, model.num_historical_steps - offset : model.num_historical_steps
    ] = wrap_angle(loc_refine_head + theta.unsqueeze(-1))[:, :offset]
    input_data["agent"]["position"] = input_position
    input_data["agent"]["heading"] = input_heading
    input_data["agent"]["velocity"] = input_v

    auto_pred = model(input_data)
    new_anchor = get_transform_mat(input_data, model)

    def get_transform_res(old_anchor, new_anchor, auto_pred):
        old_origin, old_theta, old_rot_mat = old_anchor
        new_origin, new_theta, new_rot_mat = new_anchor
        new_trans_position_propose = torch.einsum(
            "bijk,bkn->bijn",
            auto_pred["loc_propose_pos"][..., : model.output_dim],
            new_rot_mat.swapaxes(-1, -2),
        ) + new_origin[:, :2].unsqueeze(1).unsqueeze(1)
        new_pred = copy.deepcopy(auto_pred)
        new_pred["loc_propose_pos"][..., : model.output_dim] = torch.einsum(
            "bijk,bkn->bijn",
            new_trans_position_propose.cuda()
            - old_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda(),
            old_rot_mat.cuda(),
        )
        new_pred["scale_propose_pos"][..., model.output_dim - 1] = wrap_angle(
            auto_pred["scale_propose_pos"][..., model.output_dim - 1].cuda()
            + new_theta.unsqueeze(-1).unsqueeze(-1).cuda()
            - old_theta.unsqueeze(-1).unsqueeze(-1).cuda()
        )

        new_trans_position_refine = (
            torch.einsum(
                "bijk,bkn->bijn",
                auto_pred["loc_refine_pos"][..., : model.output_dim].cuda(),
                new_rot_mat.swapaxes(-1, -2).cuda(),
            )
            + new_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda()
        )
        new_pred["loc_refine_pos"][..., : model.output_dim] = torch.einsum(
            "bijk,bkn->bijn",
            new_trans_position_refine.cuda()
            - old_origin[:, :2].unsqueeze(1).unsqueeze(1).cuda(),
            old_rot_mat.cuda(),
        )
        new_pred["scale_refine_pos"][..., model.output_dim - 1] = wrap_angle(
            auto_pred["scale_refine_pos"][..., model.output_dim - 1].cuda()
            + new_theta.unsqueeze(-1).unsqueeze(-1).cuda()
            - old_theta.unsqueeze(-1).unsqueeze(-1).cuda()
        )
        return (
            new_pred,
            (new_trans_position_propose, new_trans_position_refine),
        )

    _, (new_trans_position_propose, new_trans_position_refine) = get_transform_res(
        old_anchor, new_anchor, auto_pred
    )
    if model.output_head:
        auto_traj_propose = torch.cat(
            [
                auto_pred["loc_propose_pos"][..., : model.output_dim],
                auto_pred["loc_propose_head"],
                auto_pred["scale_propose_pos"][..., : model.output_dim],
                auto_pred["conc_propose_head"],
            ],
            dim=-1,
        )
        auto_traj_refine = torch.cat(
            [
                auto_pred["loc_refine_pos"][..., : model.output_dim],
                auto_pred["loc_refine_head"],
                auto_pred["scale_refine_pos"][..., : model.output_dim],
                auto_pred["conc_refine_head"],
            ],
            dim=-1,
        )
    else:
        auto_traj_propose = torch.cat(
            [
                auto_pred["loc_propose_pos"][..., : model.output_dim],
                auto_pred["scale_propose_pos"][..., : model.output_dim],
            ],
            dim=-1,
        )
        auto_traj_refine = torch.cat(
            [
                auto_pred["loc_refine_pos"][..., : model.output_dim],
                auto_pred["scale_refine_pos"][..., : model.output_dim],
            ],
            dim=-1,
        )
    if anchor is not None:
        anchor_auto_pred, _ = get_transform_res(anchor, new_anchor, auto_pred)
        if model.output_head:
            anchor_auto_traj_propose = torch.cat(
                [
                    anchor_auto_pred["loc_propose_pos"][..., : model.output_dim],
                    anchor_auto_pred["loc_propose_head"],
                    anchor_auto_pred["scale_propose_pos"][..., : model.output_dim],
                    anchor_auto_pred["conc_propose_head"],
                ],
                dim=-1,
            )
            anchor_auto_traj_refine = torch.cat(
                [
                    anchor_auto_pred["loc_refine_pos"][..., : model.output_dim],
                    anchor_auto_pred["loc_refine_head"],
                    anchor_auto_pred["scale_refine_pos"][..., : model.output_dim],
                    anchor_auto_pred["conc_refine_head"],
                ],
                dim=-1,
            )
        else:
            anchor_auto_traj_propose = torch.cat(
                [
                    anchor_auto_pred["loc_propose_pos"][..., : model.output_dim],
                    anchor_auto_pred["scale_propose_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )
            anchor_auto_traj_refine = torch.cat(
                [
                    anchor_auto_pred["loc_refine_pos"][..., : model.output_dim],
                    anchor_auto_pred["scale_refine_pos"][..., : model.output_dim],
                ],
                dim=-1,
            )

    return (
        input_data,
        auto_pred,
        auto_traj_refine,
        auto_traj_propose,
        (new_trans_position_propose, new_trans_position_refine),
        None if anchor is None else (anchor_auto_traj_propose, anchor_auto_traj_refine),
    )


def add_new_agent(data, a, v0_x, v0_y, heading, x0, y0):
    acceleration = a
    arr_s_x = np.array([])
    arr_s_y = np.array([])
    arr_v_x = np.array([])
    arr_v_y = np.array([])
    # v0_x = 1*math.cos(1.23)
    t = 0.1
    x = x0
    y = y0
    # x0 = x = 5259.7
    # y0 = y = 318
    # x0 = x = 2665
    # y0 = y = -2410
    v_x = 0
    v_y = 0
    new_heading = torch.empty_like(data["agent"]["heading"][0])
    new_heading[:] = heading

    for i in range(110):

        a_x = acceleration * math.cos(new_heading[i])
        x = x + v0_x * t + 0.5 * a_x * (t**2)
        v0_x = v0_x + a_x * t
        v_x = v0_x
        arr_s_x = np.append(arr_s_x, x)
        arr_v_x = np.append(arr_v_x, v_x)

        a_y = acceleration * math.sin(new_heading[i])
        y = y + v0_y * t + 0.5 * a_y * (t**2)
        v0_y = v0_y + a_y * t
        v_y = v0_y
        arr_s_y = np.append(arr_s_y, y)
        arr_v_y = np.append(arr_v_y, v_y)

    new_position = torch.empty_like(data["agent"]["position"][0])
    new_position[:, 0] = torch.tensor(np.concatenate([arr_s_x]))
    new_position[:, 1] = torch.tensor(np.concatenate([arr_s_y]))

    new_velocity = torch.empty_like(data["agent"]["velocity"][0])
    new_velocity[:, 0] = torch.tensor(np.concatenate([arr_v_x]))
    new_velocity[:, 1] = torch.tensor(np.concatenate([arr_v_y]))

    data = data.clone()
    data["agent"]["num_nodes"] += 1  # num_nodes
    # av_index
    data["agent"]["valid_mask"] = torch.cat(
        [data["agent"]["valid_mask"], torch.ones_like(data["agent"]["valid_mask"][[0]])]
    )  # valid_mask
    data["agent"]["predict_mask"] = torch.cat(
        [
            data["agent"]["predict_mask"],
            torch.ones_like(data["agent"]["predict_mask"][[0]]),
        ]
    )  # predict_mask
    data["agent"]["id"][0].append(
        str(max(map(int, filter(str.isdigit, data["agent"]["id"][0]))) + 1000)
    )  # id
    data["agent"]["type"] = torch.cat(
        [data["agent"]["type"], torch.tensor([0])]
    )  # type
    data["agent"]["category"] = torch.cat(
        [data["agent"]["category"], torch.tensor([2])]
    )  # category
    data["agent"]["position"] = torch.cat(
        [data["agent"]["position"], new_position[None, :]]
    )  # position
    data["agent"]["heading"] = torch.cat(
        [data["agent"]["heading"], new_heading[None, :]]
    )  # heading
    data["agent"]["velocity"] = torch.cat(
        [data["agent"]["velocity"], new_velocity[None, :]]
    )  # velocity
    # target'
    data["agent"]["batch"] = torch.cat(
        [data["agent"]["batch"], torch.tensor([0])]
    )  # batch
    data["agent"]["ptr"][1] += 1  # ptr
    return data


def reward_function(data, new_data, model, agent_index, scenario_static_map, dataset_type='av2'):

    reward1 = 0.0; reward2 = 0.0; reward3 = 0.0; reward4 = 0.0
    
    flag = 0
    if dataset_type=='av2':
        boundary_coords = []
        for i in scenario_static_map.get_nearby_lane_segments(new_data["agent"]["position"][
                        agent_index,
                        model.num_historical_steps - 1 : model.num_historical_steps,
                        : model.output_dim,
                    ].cpu().numpy(),3):
            for j in scenario_static_map.get_lane_segment_polygon(i.id):
                    boundary_coords.append([j[0], j[1]])
            max_speed_limit_ms = 15
    
            current_velocity = new_data["agent"]["velocity"][
                agent_index,
                model.num_historical_steps - 1,
                : model.output_dim
            ].cpu().numpy()
            current_speed = np.linalg.norm(current_velocity)

            if current_speed > max_speed_limit_ms:
                reward4 -= (current_speed - max_speed_limit_ms) * 0.5 
            polygon = Polygon(boundary_coords)
            
            if polygon.contains(Point(new_data["agent"]["position"][
                        agent_index,
                        model.num_historical_steps - 1 : model.num_historical_steps,
                        : model.output_dim,
                    ].cpu().numpy())):
                flag = 1
                break
    elif dataset_type=='waymo':
        line_id = -1
        center_line = None
        center_dis = float('inf')
        point = new_data["agent"]["position"][
                        agent_index,
                        model.num_historical_steps - 1 : model.num_historical_steps,
                        : model.output_dim,
                    ].cpu().numpy()
        point = Point(point[0,0], point[0,1])
        for key, value in scenario_static_map.items():
            if value['type']=='LANE_FREEWAY' or value['type']=='LANE_SURFACE_STREET':
                centerline = LineString(value['polyline'][:,:2])
                distance_to_center = centerline.distance(point)
                if distance_to_center<center_dis:
                    center_dis=distance_to_center
                    line_id=key
                    center_line=centerline
                    
        if line_id!=-1:
            max_speed_limit_kmh = scenario_static_map[line_id]['speed_limit_kmh']
            max_speed_limit_ms = max_speed_limit_kmh / 3.6
            
            current_velocity = new_data["agent"]["velocity"][
                agent_index,
                model.num_historical_steps - 1,
                : model.output_dim
            ].cpu().numpy()
            current_speed = np.linalg.norm(current_velocity)
            if current_speed > max_speed_limit_ms:
                reward4 -= (current_speed - max_speed_limit_ms) 
            projected_dist = center_line.project(point)
            closest_point = center_line.interpolate(projected_dist)
            left_line = None
            right_line = None
            distance_to_left = float('inf')
            distance_to_right = float('inf')
            if scenario_static_map[line_id]['left_boundaries']==[]:
                center_arr = scenario_static_map[line_id]['polyline'][:,:2]
                left_bound = []
                for i in range(1, center_arr.shape[0]-1):
                    start = Point(center_arr[i-1,0], center_arr[i-1,1])
                    end = Point(center_arr[i+1,0], center_arr[i+1,1])
                    projection = Point(center_arr[i,0], center_arr[i,1]) 
                    distance_to_a = scenario_static_map[line_id]['width'][i,0]     

                    direction_vector = [end.x - start.x, end.y - start.y]
                    length = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                    unit_vector = [direction_vector[0] / length, direction_vector[1] / length]

                    a_x = projection.x - unit_vector[1] * distance_to_a
                    a_y = projection.y + unit_vector[0] * distance_to_a
                    left_bound.append([a_x, a_y])
                left_line = LineString(np.array(left_bound))
                distance_to_left = left_line.distance(point)

            if scenario_static_map[line_id]['right_boundaries']==[]:
                center_arr = scenario_static_map[line_id]['polyline'][:,:2]
                right_bound = []
                for i in range(1, center_arr.shape[0]-1):
                    start = Point(center_arr[i-1,0], center_arr[i-1,1])
                    end = Point(center_arr[i+1,0], center_arr[i+1,1])
                    projection = Point(center_arr[i,0], center_arr[i,1]) 
                    distance_to_a = scenario_static_map[line_id]['width'][i,1]     

                    direction_vector = [end.x - start.x, end.y - start.y]
                    length = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                    unit_vector = [direction_vector[0] / length, direction_vector[1] / length]

                    a_x = projection.x + unit_vector[1] * distance_to_a
                    a_y = projection.y - unit_vector[0] * distance_to_a
                    right_bound.append([a_x, a_y])
                right_line = LineString(np.array(right_bound))
                distance_to_left = right_line.distance(point)
            for line in scenario_static_map[line_id]['left_boundaries']:
                left_boundary = LineString(scenario_static_map[line['boundary_feature_id']]['polyline'][:,:2])
                dis_left = left_boundary.distance(point)
                if dis_left<distance_to_left:
                    distance_to_left = dis_left
                    left_line = left_boundary
            for line in scenario_static_map[line_id]['right_boundaries']:
                right_boundary = LineString(scenario_static_map[line['boundary_feature_id']]['polyline'][:,:2])
                dis_right = right_boundary.distance(point)
                if dis_right<distance_to_right:
                    distance_to_right = dis_right
                    right_line = right_boundary
            if left_line==None and right_line==None:
                print("without boundaries!")
                width = 0
            elif left_line!=None and right_line==None:
                left_distance = left_line.distance(closest_point)
                index = (np.abs(scenario_static_map[line_id]['width'][:,1] - left_distance)).argmin()
                width = left_distance+scenario_static_map[line_id]['width'][index,1]
            elif left_line==None and right_line!=None:
                right_distance = right_line.distance(closest_point)
                index = (np.abs(scenario_static_map[line_id]['width'][:,0] - right_distance)).argmin()
                width = right_distance+scenario_static_map[line_id]['width'][index,0]
                distance_to_left = 0
            else:
                width = left_line.distance(closest_point)+right_line.distance(closest_point)
            if distance_to_left>0 and distance_to_right>0 and distance_to_left<width and distance_to_right<width:
                flag = 1
    if not flag:
        reward3 = -10

    gt = data["agent"]["position"][agent_index, -1, : model.output_dim]
    start_point = data["agent"]["position"][
        agent_index, model.num_historical_steps - 1, : model.output_dim
    ]
    current_position = new_data["agent"]["position"][
        agent_index, model.num_historical_steps - 1, : model.output_dim
    ]
    l1_norm_current_distance = torch.norm(current_position - gt, p=1, dim=-1)
    reward1 = -l1_norm_current_distance/10

    for i in range(new_data['agent']['num_nodes']):
        if i != agent_index:
 
            distance = torch.norm(
                new_data["agent"]["position"][
                    agent_index,
                    model.num_historical_steps - 1 : model.num_historical_steps,
                    : model.output_dim,
                ]
                - new_data["agent"]["position"][
                    i,
                    model.num_historical_steps - 1 : model.num_historical_steps,
                    : model.output_dim,
                ],
                p=2,
                dim=-1,
            )
            if distance < 2:
                reward2 -= 10
                break
    # for i in range(new_data['agent']['num_nodes']):
    #     if i != agent_index:

    #         heading = new_data["agent"]["heading"][agent_index,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()
    #         vehicle_length = new_data["agent"]["length"][agent_index,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()
    #         vehicle_width = new_data["agent"]["width"][agent_index,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()

    #         other_heading = new_data["agent"]["heading"][i,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()
    #         other_length = new_data["agent"]["length"][i,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()
    #         other_width = new_data["agent"]["width"][i,model.num_historical_steps - 1 : model.num_historical_steps].cpu().numpy()
            
    #         current_position = new_data["agent"]["position"][
    #             agent_index,
    #             model.num_historical_steps - 1 : model.num_historical_steps,
    #             : model.output_dim,
    #         ].cpu().numpy().flatten()
            
    #         dx = vehicle_length / 2 * np.cos(heading)
    #         dy = vehicle_length / 2 * np.sin(heading)
    #         corner1 = (current_position[0] - dx + vehicle_width / 2 * np.sin(heading), current_position[1] - dy - vehicle_width / 2 * np.cos(heading))
    #         corner2 = (current_position[0] - dx - vehicle_width / 2 * np.sin(heading), current_position[1] - dy + vehicle_width / 2 * np.cos(heading))
    #         corner3 = (current_position[0] + dx - vehicle_width / 2 * np.sin(heading), current_position[1] + dy + vehicle_width / 2 * np.cos(heading))
    #         corner4 = (current_position[0] + dx + vehicle_width / 2 * np.sin(heading), current_position[1] + dy - vehicle_width / 2 * np.cos(heading))
    #         vehicle_polygon = Polygon([corner1, corner2, corner3, corner4])

    #         # Calculate the four corners of the other vehicle's bounding box
    #         other_position = new_data["agent"]["position"][
    #             i,
    #             model.num_historical_steps - 1 : model.num_historical_steps,
    #             : model.output_dim,
    #         ].cpu().numpy().flatten()
            
    #         other_dx = other_length / 2 * np.cos(other_heading)
    #         other_dy = other_length / 2 * np.sin(other_heading)
    #         other_corner1 = (other_position[0] - other_dx + other_width / 2 * np.sin(other_heading), other_position[1] - other_dy - other_width / 2 * np.cos(other_heading))
    #         other_corner2 = (other_position[0] - other_dx - other_width / 2 * np.sin(other_heading), other_position[1] - other_dy + other_width / 2 * np.cos(other_heading))
    #         other_corner3 = (other_position[0] + other_dx - other_width / 2 * np.sin(other_heading), other_position[1] + other_dy + other_width / 2 * np.cos(other_heading))
    #         other_corner4 = (other_position[0] + other_dx + other_width / 2 * np.sin(other_heading), other_position[1] + other_dy - other_width / 2 * np.cos(other_heading))
    #         other_vehicle_polygon = Polygon([other_corner1, other_corner2, other_corner3, other_corner4])

    #         # Check for intersection (collision) between the two vehicle polygons
    #         if vehicle_polygon.intersects(other_vehicle_polygon):
    #             reward2 -= 10  
    #             break

    return reward1 + reward2 + reward3 + reward4


def cost_function(new_data, model, agent_index, distance_limit, choose_agent):
    cost_value = 0
    
    min_distance = 9999
    for i in range(len(choose_agent)):
        if choose_agent[i] != agent_index:
            distance = torch.norm(
                new_data["agent"]["position"][
                    agent_index,
                    model.num_historical_steps - 1 : model.num_historical_steps,
                    : model.output_dim,
                ]
                - new_data["agent"]["position"][
                    choose_agent[i],
                    model.num_historical_steps - 1 : model.num_historical_steps,
                    : model.output_dim,
                ],
                p=2,
                dim=-1,
            )
            if distance < min_distance:
                min_distance = distance
                
    if min_distance < distance_limit:
        cost_value = 1

    return cost_value

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_advantage(
            gae: bool,
            td_delta: torch.Tensor,
            device,
            dones: torch.Tensor,
            gamma: float,
            gae_lambda: float,
        ):
 
        if gae:
            advantage = 0
            advantage_list = []
            td_delta = td_delta.cpu().numpy()
            for t in reversed(range(len(td_delta))):
                advantage = (
                    gamma * gae_lambda * advantage * (1 - dones[t].cpu())
                    + td_delta[t]
                )
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantages = (
                torch.tensor(advantage_list, dtype=torch.float)
                .to(device)
                .reshape(-1, 1)
            )
        else:
            advantages = td_delta

        return advantages
