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

import math
import os
import pickle
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
import torch
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import angle_between_2d_vectors
from utils import safe_list_index
from utils import side_to_directed_lineseg

try:
    from av2.geometry.interpolate import interp_arc
    from waymo_open_dataset.protos import map_pb2
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    interp_arc = object
    map_pb2 = object
    scenario_pb2 = object

tf.config.set_visible_devices([], 'GPU')


class WaymoDataset(Dataset):
    """Dataset class for Waymo Open Motion Dataset v1.2.0.

    See https://waymo.com/open/data/motion for more information about the dataset.

    Args:
        root (string): the root folder of the dataset.
        split (string): specify the split of the dataset: `"train"` | `"val"` | `"test"`.
        interactive (boolean, Optional): if True, use the interactive split of the validation/test set. (default: False)
        raw_dir (string, optional): optionally specify the directory of the raw data. By default, the raw directory is
            path/to/root/split/raw. If specified, the path of the raw tfrecord files is path/to/raw_dir/*.
            (default: None)
        processed_dir (string, optional): optionally specify the directory of the processed data. By default, the
            processed directory is path/to/root/split/processed/. If specified, the path of the processed .pkl files is
            path/to/processed_dir/*.pkl. If all .pkl files exist in the processed directory, data preprocessing will be
            skipped. (default: None)
        transform (callable, optional): a function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access. (default: None)
        dim (int, Optional): 2D or 3D data. (default: 3)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 11)
        num_future_steps (int, Optional): the number of future time steps. (default: 80)
        predict_unseen_agents (boolean, Optional): if False, filter out agents that are unseen during the historical
            time steps. (default: False)
        vector_repr (boolean, Optional): if True, a time step t is valid only when both t and t-1 are valid.
            (default: True)
        resolution_meters (float, Optional): the resolution of HD map's sampling distance in meters. (default: 1.0)
        traffic_signal (boolean, Optional): whether the data include information about traffic signals. (default: True)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 interactive: bool = False,
                 dim: int = 3,
                 num_historical_steps: int = 11,
                 num_future_steps: int = 80,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 resolution_meters: float = 1.0,
                 traffic_signal: bool = True,
                 **kwargs) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split
        self.dir = {
            'train': 'training',
            'val': 'validation',
            'test': 'testing',
        }[split]

        if raw_dir is None:
            raw_dir = os.path.join(root, self.dir, 'raw')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isfile(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isfile(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, self.dir, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self.resolution_meters = resolution_meters
        self.traffic_signal = traffic_signal
        self._num_raw_files = {
            'train': 1000,
            'val': 150,
            'test': 150,
        }[split]
        self._num_samples = {
            'train': 486995,
            # 'val': 44097 if not interactive else 43479,
            # 'test': 44920 if not interactive else 44154,
            'val': 44097,
            'test': 44920,
        }[split]
        self._lane_type_dict = {
            map_pb2.LaneCenter.TYPE_UNDEFINED: 'UNDEFINED',
            map_pb2.LaneCenter.TYPE_FREEWAY: 'FREEWAY',
            map_pb2.LaneCenter.TYPE_SURFACE_STREET: 'SURFACE_STREET',
            map_pb2.LaneCenter.TYPE_BIKE_LANE: 'BIKE_LANE',
        }
        self._road_edge_type_dict = {
            map_pb2.RoadEdge.TYPE_UNKNOWN: 'ROAD_EDGE_UNKNOWN',
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: 'ROAD_EDGE_BOUNDARY',
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: 'ROAD_EDGE_MEDIAN',
        }
        self._road_line_type_dict = {
            map_pb2.RoadLine.TYPE_UNKNOWN: 'ROAD_LINE_UNKNOWN',
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
            map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: 'ROAD_LINE_SOLID_DOUBLE_WHITE',
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
            map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
            map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
            map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW',
        }
        self._traffic_signal_state_type_dict = {
            map_pb2.TrafficSignalLaneState.LANE_STATE_UNKNOWN: 'UNKNOWN',
            map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: 'ARROW_STOP',
            map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: 'ARROW_CAUTION',
            map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: 'ARROW_GO',
            map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: 'STOP',
            map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: 'CAUTION',
            map_pb2.TrafficSignalLaneState.LANE_STATE_GO: 'GO',
            map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: 'FLASHING_STOP',
            map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION: 'FLASHING_CAUTION',
        }
        self._agent_types = ['UNSET', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'OTHER']
        self._polygon_types = ['UNDEFINED', 'FREEWAY', 'SURFACE_STREET', 'BIKE_LANE', 'CROSSWALK', 'SPEED_BUMP',
                               'DRIVEWAY']
        self._point_types = ['ROAD_EDGE_UNKNOWN', 'ROAD_EDGE_BOUNDARY', 'ROAD_EDGE_MEDIAN', 'ROAD_LINE_UNKNOWN',
                             'ROAD_LINE_BROKEN_SINGLE_WHITE', 'ROAD_LINE_SOLID_SINGLE_WHITE',
                             'ROAD_LINE_SOLID_DOUBLE_WHITE', 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
                             'ROAD_LINE_BROKEN_DOUBLE_YELLOW', 'ROAD_LINE_SOLID_SINGLE_YELLOW',
                             'ROAD_LINE_SOLID_DOUBLE_YELLOW', 'ROAD_LINE_PASSING_DOUBLE_YELLOW', 'CENTERLINE',
                             'CROSSWALK', 'SPEED_BUMP', 'DRIVEWAY']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
        super(WaymoDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):
            records = tf.data.TFRecordDataset(os.path.join(self.raw_dir, raw_file_name))
            for record in records:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                self._processed_file_names.append(f'{scenario.scenario_id}.pkl')
                data = dict()
                data['scenario_id'] = scenario.scenario_id
                data['agent'] = self.get_agent_features(scenario)
                data.update(self.get_map_features(scenario))
                with open(os.path.join(self.processed_dir, f'{scenario.scenario_id}.pkl'), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_agent_features(self, scenario) -> Dict[str, Any]:
        agent_ids = [track.id for track in scenario.tracks]
        if not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            interest_agent_ids = []
            for track in scenario.tracks:
                for state in track.states[:self.num_historical_steps]:
                    if state.valid:
                        interest_agent_ids.append(track.id)
                        break
        else:
            interest_agent_ids = agent_ids

        num_agents = len(interest_agent_ids)
        av_idx = interest_agent_ids.index(agent_ids[scenario.sdc_track_index])

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        target_mask = torch.zeros(num_agents, dtype=torch.bool)
        interact_mask = torch.zeros(num_agents, dtype=torch.bool)
        agent_id = torch.zeros(num_agents, dtype=torch.int32)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        length = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        width = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        height = torch.zeros(num_agents, self.num_steps, dtype=torch.float)

        for track in scenario.tracks:
            if track.id not in interest_agent_ids:
                continue
            agent_idx = interest_agent_ids.index(track.id)
            agent_steps = [t for t, state in enumerate(track.states) if state.valid]

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track.id
            agent_type[agent_idx] = track.object_type
            num_states = len(track.states)
            position[agent_idx, :num_states, 0] = torch.tensor([state.center_x for state in track.states],
                                                               dtype=torch.float)
            position[agent_idx, :num_states, 1] = torch.tensor([state.center_y for state in track.states],
                                                               dtype=torch.float)
            if self.dim == 3:
                position[agent_idx, :num_states, 2] = torch.tensor([state.center_z for state in track.states],
                                                                   dtype=torch.float)
            heading[agent_idx, :num_states] = torch.tensor([state.heading for state in track.states], dtype=torch.float)
            velocity[agent_idx, :num_states, 0] = torch.tensor([state.velocity_x for state in track.states],
                                                               dtype=torch.float)
            velocity[agent_idx, :num_states, 1] = torch.tensor([state.velocity_y for state in track.states],
                                                               dtype=torch.float)
            length[agent_idx, :num_states] = torch.tensor([abs(state.length) for state in track.states],
                                                          dtype=torch.float)
            width[agent_idx, :num_states] = torch.tensor([abs(state.width) for state in track.states],
                                                         dtype=torch.float)
            height[agent_idx, :num_states] = torch.tensor([abs(state.height) for state in track.states],
                                                          dtype=torch.float)

        for track_to_predict in scenario.tracks_to_predict:
            target_mask[interest_agent_ids.index(agent_ids[track_to_predict.track_index])] = True
        for object_of_interest in scenario.objects_of_interest:
            interact_mask[interest_agent_ids.index(object_of_interest)] = True

        if self.split == 'test':
            predict_mask[current_valid_mask | target_mask, self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'target_mask': target_mask,
            'interact_mask': interact_mask,
            'id': agent_id,
            'type': agent_type,
            'position': position,
            'heading': heading,
            'velocity': velocity,
            'length': length[:, :self.num_historical_steps].clone().detach(),
            'width': width[:, :self.num_historical_steps].clone().detach(),
            'height': height[:, :self.num_historical_steps].clone().detach(),
        }

    def get_map_features(self, scenario) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        lane_ids, crosswalk_ids, speed_bump_ids, driveway_ids = [], [], [], []
        for map_feature in scenario.map_features:
            feature_type = map_feature.WhichOneof('feature_data')
            if feature_type == 'lane':
                lane = getattr(map_feature, 'lane')
                if len(lane.polyline) > 1:
                    lane_ids.append(map_feature.id)
            elif feature_type == 'crosswalk':
                crosswalk = getattr(map_feature, 'crosswalk')
                if len(crosswalk.polygon) > 2:
                    crosswalk_ids.append(map_feature.id)
            elif feature_type == 'speed_bump':
                speed_bump = getattr(map_feature, 'speed_bump')
                if len(speed_bump.polygon) > 2:
                    speed_bump_ids.append(map_feature.id)
            elif feature_type == 'driveway':
                driveway = getattr(map_feature, 'driveway')
                if len(driveway.polygon) > 2:
                    driveway_ids.append(map_feature.id)
            else:
                continue
        bi_polygon_ids = crosswalk_ids + speed_bump_ids + driveway_ids
        num_bi_polygons = len(bi_polygon_ids)
        polygon_ids = lane_ids + bi_polygon_ids
        num_polygons = len(lane_ids) + num_bi_polygons * 2

        # initialization
        lanes, boundaries, stop_signs, crosswalks, speed_bumps, driveways = (dict(), dict(), dict(), dict(), dict(),
                                                                             dict())
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_speed_limit = torch.zeros(num_polygons, dtype=torch.float)
        polygon_stop_sign_mask = torch.zeros(num_polygons, dtype=torch.bool)
        polygon_stop_sign_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_trafic_signal_mask = torch.zeros(num_polygons, self.num_steps, dtype=torch.bool)
        polygon_traffic_signal_state = torch.zeros(num_polygons, self.num_steps, dtype=torch.uint8)
        polygon_stop_point = torch.zeros(num_polygons, self.num_steps, self.dim, dtype=torch.float)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        feature_type_dict = {
            'lane': lanes,
            'road_line': boundaries,
            'road_edge': boundaries,
            'stop_sign': stop_signs,
            'crosswalk': crosswalks,
            'speed_bump': speed_bumps,
            'driveway': driveways,
        }
        for map_feature in scenario.map_features:
            feature_type = map_feature.WhichOneof('feature_data')
            feature_type_dict[feature_type][map_feature.id] = getattr(map_feature, feature_type)

        for lane_id, lane in lanes.items():
            lane_idx = safe_list_index(polygon_ids, lane_id)
            if lane_idx is None:
                continue
            raw_centerline = torch.tensor([[point.x, point.y, point.z] for point in lane.polyline], dtype=torch.float)
            step_size = math.floor(self.resolution_meters / 0.5)
            sample_inds = torch.arange(0, raw_centerline.size(0), step_size)
            if (raw_centerline.size(0) - 1) % step_size != 0:
                sample_inds = torch.cat([sample_inds, torch.tensor([raw_centerline.size(0) - 1])], dim=0)
            centerline = raw_centerline[sample_inds]
            polygon_position[lane_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                        centerline[1, 0] - centerline[0, 0])
            polygon_type[lane_idx] = self._polygon_types.index(self._lane_type_dict[lane.type])
            polygon_speed_limit[lane_idx] = lane.speed_limit_mph

            left_boundary, right_boundary, left_type, right_type = [], [], [], []
            for boundary in lane.left_boundaries:
                if boundary.boundary_feature_id not in boundaries:
                    continue
                left_polyline = torch.tensor([[point.x, point.y, point.z] for point in
                                              boundaries[boundary.boundary_feature_id].polyline], dtype=torch.float)
                if angle_between_2d_vectors(
                        ctr_vector=raw_centerline[boundary.lane_end_index, :2] -
                                   raw_centerline[boundary.lane_start_index, :2],
                        nbr_vector=left_polyline[-1, :2] -
                                   left_polyline[0, :2]).abs() > math.pi / 2:
                    left_polyline = left_polyline.flip(dims=[0])
                step_size = math.floor(self.resolution_meters / 0.5)
                sample_inds = torch.arange(0, left_polyline.size(0), step_size)
                if (left_polyline.size(0) - 1) % step_size != 0:
                    sample_inds = torch.cat([sample_inds, torch.tensor([left_polyline.size(0) - 1])], dim=0)
                left_polyline = left_polyline[sample_inds]
                left_boundary.append(left_polyline)
                if boundary.boundary_type == map_pb2.RoadLine.TYPE_UNKNOWN:
                    boundary_type = self._point_types.index(
                        self._road_edge_type_dict[boundaries[boundary.boundary_feature_id].type])
                else:
                    boundary_type = self._point_types.index(
                        self._road_line_type_dict[boundaries[boundary.boundary_feature_id].type])
                left_type.extend([boundary_type] * (left_polyline.size(0) - 1))
            for boundary in lane.right_boundaries:
                if boundary.boundary_feature_id not in boundaries:
                    continue
                right_polyline = torch.tensor([[point.x, point.y, point.z] for point in
                                               boundaries[boundary.boundary_feature_id].polyline], dtype=torch.float)
                if angle_between_2d_vectors(
                        ctr_vector=raw_centerline[boundary.lane_end_index, :2] -
                                   raw_centerline[boundary.lane_start_index, :2],
                        nbr_vector=right_polyline[-1, :2] -
                                   right_polyline[0, :2]).abs() > math.pi / 2:
                    right_polyline = right_polyline.flip(dims=[0])
                step_size = math.floor(self.resolution_meters / 0.5)
                sample_inds = torch.arange(0, right_polyline.size(0), step_size)
                if (right_polyline.size(0) - 1) % step_size != 0:
                    sample_inds = torch.cat([sample_inds, torch.tensor([right_polyline.size(0) - 1])], dim=0)
                right_polyline = right_polyline[sample_inds]
                right_boundary.append(right_polyline)
                if boundary.boundary_type == map_pb2.RoadLine.TYPE_UNKNOWN:
                    boundary_type = self._point_types.index(
                        self._road_edge_type_dict[boundaries[boundary.boundary_feature_id].type])
                else:
                    boundary_type = self._point_types.index(
                        self._road_line_type_dict[boundaries[boundary.boundary_feature_id].type])
                right_type.extend([boundary_type] * (right_polyline.size(0) - 1))
            if len(left_boundary) == 0:
                left_vectors = torch.zeros(0, 3, dtype=torch.float)
                left_boundary = torch.zeros(0, 3, dtype=torch.float)
            else:
                left_vectors = []
                for i in range(len(left_boundary)):
                    left_vectors.append(left_boundary[i][1:] - left_boundary[i][:-1])
                    left_boundary[i] = left_boundary[i][:-1]
                left_vectors = torch.cat(left_vectors, dim=0)
                left_boundary = torch.cat(left_boundary, dim=0)
            if len(right_boundary) == 0:
                right_vectors = torch.zeros(0, 3, dtype=torch.float)
                right_boundary = torch.zeros(0, 3, dtype=torch.float)
            else:
                right_vectors = []
                for i in range(len(right_boundary)):
                    right_vectors.append(right_boundary[i][1:] - right_boundary[i][:-1])
                    right_boundary[i] = right_boundary[i][:-1]
                right_vectors = torch.cat(right_vectors, dim=0)
                right_boundary = torch.cat(right_boundary, dim=0)
            center_vectors = centerline[1:] - centerline[:-1]
            centerline = centerline[:-1]
            point_position[lane_idx] = torch.cat([left_boundary[:, :self.dim],
                                                  right_boundary[:, :self.dim],
                                                  centerline[:, :self.dim]], dim=0)
            point_orientation[lane_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                     torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                     torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_magnitude[lane_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                              right_vectors[:, :2],
                                                              center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]], dim=0)
            left_type = torch.tensor(left_type, dtype=torch.uint8)
            right_type = torch.tensor(right_type, dtype=torch.uint8)
            center_type = torch.full((len(center_vectors),), self._point_types.index('CENTERLINE'), dtype=torch.uint8)
            point_type[lane_idx] = torch.cat([left_type, right_type, center_type], dim=0)
            point_side[lane_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk_id, crosswalk in crosswalks.items():
            crosswalk_idx = safe_list_index(polygon_ids, crosswalk_id)
            if crosswalk_idx is None:
                continue
            crosswalk_polygon = Polygon([[point.x, point.y, point.z] for point in crosswalk.polygon])
            crosswalk_polygon = orient(crosswalk_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(crosswalk_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            bbox = torch.tensor(list(crosswalk_polygon.minimum_rotated_rectangle.exterior.coords), dtype=torch.float)
            if torch.norm(bbox[0] - bbox[1], p=2, dim=-1) > torch.norm(bbox[1] - bbox[2], p=2, dim=-1):
                start_position = (bbox[0] + bbox[3]) / 2
                end_position = (bbox[1] + bbox[2]) / 2
            else:
                start_position = (bbox[0] + bbox[1]) / 2
                end_position = (bbox[2] + bbox[3]) / 2
            intersect_linestring = crosswalk_polygon.intersection(
                LineString(torch.stack([start_position, end_position], dim=0)))
            if isinstance(intersect_linestring, LineString):
                start_position = torch.tensor(intersect_linestring.coords[0], dtype=torch.float)
                end_position = torch.tensor(intersect_linestring.coords[-1], dtype=torch.float)
            else:
                intersect_positions = []
                for geom in intersect_linestring.geoms:
                    intersect_positions.append(torch.tensor(list(geom.coords), dtype=torch.float))
                intersect_positions = torch.cat(intersect_positions, dim=0)
                start_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                                start_position.unsqueeze(0), p=2, dim=-1).argmin()]
                end_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                              end_position.unsqueeze(0), p=2, dim=-1).argmin()]
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + num_bi_polygons] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + num_bi_polygons] = torch.atan2((start_position - end_position)[1],
                                                                               (start_position - end_position)[0])
            polygon_type[crosswalk_idx] = self._polygon_types.index('CROSSWALK')
            polygon_type[crosswalk_idx + num_bi_polygons] = self._polygon_types.index('CROSSWALK')
            polygon_speed_limit[crosswalk_idx] = 0.0
            polygon_speed_limit[crosswalk_idx + num_bi_polygons] = 0.0

            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() /
                                              self.resolution_meters) + 1
            centerline = torch.from_numpy(interp_arc(int(num_centerline_points),
                                                     torch.stack([start_position, end_position],
                                                                 dim=0).numpy())).float()
            point_position[crosswalk_idx] = torch.cat([boundary[:-1, :self.dim], centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + num_bi_polygons] = torch.cat([boundary[:-1, :self.dim],
                                                                         centerline.flip(dims=[0])[:-1, :self.dim]],
                                                                        dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + num_bi_polygons] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([boundary_vectors[:, :2],
                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + num_bi_polygons] = torch.norm(
                torch.cat([boundary_vectors[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([boundary_vectors[:, 2], center_vectors[:, 2]], dim=0)
            point_height[crosswalk_idx + num_bi_polygons] = torch.cat([boundary_vectors[:, 2],
                                                                       -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(boundary_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + num_bi_polygons] = torch.cat(
                [torch.full((len(boundary_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, start_position, end_position)))
            point_side[crosswalk_idx] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, end_position, start_position)))
            point_side[crosswalk_idx + num_bi_polygons] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for speed_bump_id, speed_bump in speed_bumps.items():
            speed_bump_idx = safe_list_index(polygon_ids, speed_bump_id)
            if speed_bump_idx is None:
                continue
            speed_bump_polygon = Polygon([[point.x, point.y, point.z] for point in speed_bump.polygon])
            speed_bump_polygon = orient(speed_bump_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(speed_bump_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            bbox = torch.tensor(list(speed_bump_polygon.minimum_rotated_rectangle.exterior.coords), dtype=torch.float)
            if torch.norm(bbox[0] - bbox[1], p=2, dim=-1) > torch.norm(bbox[1] - bbox[2], p=2, dim=-1):
                start_position = (bbox[0] + bbox[3]) / 2
                end_position = (bbox[1] + bbox[2]) / 2
            else:
                start_position = (bbox[0] + bbox[1]) / 2
                end_position = (bbox[2] + bbox[3]) / 2
            intersect_linestring = speed_bump_polygon.intersection(
                LineString(torch.stack([start_position, end_position], dim=0)))
            if isinstance(intersect_linestring, LineString):
                start_position = torch.tensor(intersect_linestring.coords[0], dtype=torch.float)
                end_position = torch.tensor(intersect_linestring.coords[-1], dtype=torch.float)
            else:
                intersect_positions = []
                for geom in intersect_linestring.geoms:
                    intersect_positions.append(torch.tensor(list(geom.coords), dtype=torch.float))
                intersect_positions = torch.cat(intersect_positions, dim=0)
                start_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                                start_position.unsqueeze(0), p=2, dim=-1).argmin()]
                end_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                              end_position.unsqueeze(0), p=2, dim=-1).argmin()]
            polygon_position[speed_bump_idx] = start_position[:self.dim]
            polygon_position[speed_bump_idx + num_bi_polygons] = end_position[:self.dim]
            polygon_orientation[speed_bump_idx] = torch.atan2((end_position - start_position)[1],
                                                              (end_position - start_position)[0])
            polygon_orientation[speed_bump_idx + num_bi_polygons] = torch.atan2((start_position - end_position)[1],
                                                                                (start_position - end_position)[0])
            polygon_type[speed_bump_idx] = self._polygon_types.index('SPEED_BUMP')
            polygon_type[speed_bump_idx + num_bi_polygons] = self._polygon_types.index('SPEED_BUMP')
            polygon_speed_limit[speed_bump_idx] = 0.0
            polygon_speed_limit[speed_bump_idx + num_bi_polygons] = 0.0

            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() /
                                              self.resolution_meters) + 1
            centerline = torch.from_numpy(interp_arc(int(num_centerline_points),
                                                     torch.stack([start_position, end_position],
                                                                 dim=0).numpy())).float()
            point_position[speed_bump_idx] = torch.cat([boundary[:-1, :self.dim], centerline[:-1, :self.dim]], dim=0)
            point_position[speed_bump_idx + num_bi_polygons] = torch.cat([boundary[:-1, :self.dim],
                                                                          centerline.flip(dims=[0])[:-1, :self.dim]],
                                                                         dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[speed_bump_idx] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[speed_bump_idx + num_bi_polygons] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[speed_bump_idx] = torch.norm(torch.cat([boundary_vectors[:, :2],
                                                                    center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[speed_bump_idx + num_bi_polygons] = torch.norm(
                torch.cat([boundary_vectors[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[speed_bump_idx] = torch.cat([boundary_vectors[:, 2], center_vectors[:, 2]], dim=0)
            point_height[speed_bump_idx + num_bi_polygons] = torch.cat([boundary_vectors[:, 2],
                                                                        -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            speed_bump_type = self._point_types.index('SPEED_BUMP')
            center_type = self._point_types.index('CENTERLINE')
            point_type[speed_bump_idx] = torch.cat([
                torch.full((len(boundary_vectors),), speed_bump_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[speed_bump_idx + num_bi_polygons] = torch.cat(
                [torch.full((len(boundary_vectors),), speed_bump_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, start_position, end_position)))
            point_side[speed_bump_idx] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, end_position, start_position)))
            point_side[speed_bump_idx + num_bi_polygons] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for driveway_id, driveway in driveways.items():
            driveway_idx = safe_list_index(polygon_ids, driveway_id)
            if driveway_idx is None:
                continue
            driveway_polygon = Polygon([[point.x, point.y, point.z] for point in driveway.polygon])
            driveway_polygon = orient(driveway_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(driveway_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            bbox = torch.tensor(list(driveway_polygon.minimum_rotated_rectangle.exterior.coords), dtype=torch.float)
            if torch.norm(bbox[0] - bbox[1], p=2, dim=-1) > torch.norm(bbox[1] - bbox[2], p=2, dim=-1):
                start_position = (bbox[0] + bbox[3]) / 2
                end_position = (bbox[1] + bbox[2]) / 2
            else:
                start_position = (bbox[0] + bbox[1]) / 2
                end_position = (bbox[2] + bbox[3]) / 2
            intersect_linestring = driveway_polygon.intersection(
                LineString(torch.stack([start_position, end_position], dim=0)))
            if isinstance(intersect_linestring, LineString):
                start_position = torch.tensor(intersect_linestring.coords[0], dtype=torch.float)
                end_position = torch.tensor(intersect_linestring.coords[-1], dtype=torch.float)
            else:
                intersect_positions = []
                for geom in intersect_linestring.geoms:
                    intersect_positions.append(torch.tensor(list(geom.coords), dtype=torch.float))
                intersect_positions = torch.cat(intersect_positions, dim=0)
                start_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                                start_position.unsqueeze(0), p=2, dim=-1).argmin()]
                end_position = intersect_positions[torch.norm(intersect_positions[:, :2] -
                                                              end_position.unsqueeze(0), p=2, dim=-1).argmin()]
            polygon_position[driveway_idx] = start_position[:self.dim]
            polygon_position[driveway_idx + num_bi_polygons] = end_position[:self.dim]
            polygon_orientation[driveway_idx] = torch.atan2((end_position - start_position)[1],
                                                            (end_position - start_position)[0])
            polygon_orientation[driveway_idx + num_bi_polygons] = torch.atan2((start_position - end_position)[1],
                                                                              (start_position - end_position)[0])
            polygon_type[driveway_idx] = self._polygon_types.index('DRIVEWAY')
            polygon_type[driveway_idx + num_bi_polygons] = self._polygon_types.index('DRIVEWAY')
            polygon_speed_limit[driveway_idx] = 0.0
            polygon_speed_limit[driveway_idx + num_bi_polygons] = 0.0

            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() /
                                              self.resolution_meters) + 1
            centerline = torch.from_numpy(interp_arc(int(num_centerline_points),
                                                     torch.stack([start_position, end_position],
                                                                 dim=0).numpy())).float()
            point_position[driveway_idx] = torch.cat([boundary[:-1, :self.dim], centerline[:-1, :self.dim]], dim=0)
            point_position[driveway_idx + num_bi_polygons] = torch.cat([boundary[:-1, :self.dim],
                                                                        centerline.flip(dims=[0])[:-1, :self.dim]],
                                                                       dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[driveway_idx] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[driveway_idx + num_bi_polygons] = torch.cat(
                [torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[driveway_idx] = torch.norm(torch.cat([boundary_vectors[:, :2],
                                                                  center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[driveway_idx + num_bi_polygons] = torch.norm(
                torch.cat([boundary_vectors[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[driveway_idx] = torch.cat([boundary_vectors[:, 2], center_vectors[:, 2]], dim=0)
            point_height[driveway_idx + num_bi_polygons] = torch.cat([boundary_vectors[:, 2],
                                                                      -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            driveway_type = self._point_types.index('DRIVEWAY')
            center_type = self._point_types.index('CENTERLINE')
            point_type[driveway_idx] = torch.cat([
                torch.full((len(boundary_vectors),), driveway_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[driveway_idx + num_bi_polygons] = torch.cat(
                [torch.full((len(boundary_vectors),), driveway_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, start_position, end_position)))
            point_side[driveway_idx] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            boundary_sides = []
            for boundary_point in boundary[:-1]:
                boundary_sides.append(
                    self._point_sides.index(side_to_directed_lineseg(boundary_point, end_position, start_position)))
            point_side[driveway_idx + num_bi_polygons] = torch.cat(
                [torch.tensor(boundary_sides, dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for stop_sign in stop_signs.values():
            for lane_id in stop_sign.lane:
                lane_idx = safe_list_index(polygon_ids, lane_id)
                if lane_idx is None:
                    continue
                polygon_stop_sign_mask[lane_idx] = True
                polygon_stop_sign_position[lane_idx, 0] = stop_sign.position.x
                polygon_stop_sign_position[lane_idx, 1] = stop_sign.position.y
                if self.dim == 3:
                    polygon_stop_sign_position[lane_idx, 2] = stop_sign.position.z

        for t, dynamic_map_state in enumerate(scenario.dynamic_map_states):
            for traffic_signal_state in dynamic_map_state.lane_states:
                lane_id = traffic_signal_state.lane
                lane_idx = safe_list_index(polygon_ids, lane_id)
                if lane_idx is None:
                    continue
                polygon_trafic_signal_mask[lane_idx, t] = True
                polygon_traffic_signal_state[lane_idx, t] = traffic_signal_state.state
                polygon_stop_point[lane_idx, t, 0] = traffic_signal_state.stop_point.x
                polygon_stop_point[lane_idx, t, 1] = traffic_signal_state.stop_point.y
                if self.dim == 3:
                    polygon_stop_point[lane_idx, t, 2] = traffic_signal_state.stop_point.z

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index, polygon_to_polygon_type = [], []
        for lane_id, lane in lanes.items():
            lane_idx = safe_list_index(polygon_ids, lane_id)
            if lane_idx is None:
                continue
            pred_inds = []
            for pred in lane.entry_lanes:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in lane.exit_lanes:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            left_inds = []
            for left in lane.left_neighbors:
                left_idx = safe_list_index(polygon_ids, left.feature_id)
                if left_idx is not None:
                    left_inds.append(left_idx)
            if len(left_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(left_inds, dtype=torch.long),
                                 torch.full((len(left_inds),), lane_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(left_inds),), self._polygon_to_polygon_types.index('LEFT'), dtype=torch.uint8))
            right_inds = []
            for right in lane.right_neighbors:
                right_idx = safe_list_index(polygon_ids, right.feature_id)
                if right_idx is not None:
                    right_inds.append(right_idx)
            if len(right_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(right_inds, dtype=torch.long),
                                 torch.full((len(right_inds),), lane_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(right_inds),), self._polygon_to_polygon_types.index('RIGHT'), dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['speed_limit'] = polygon_speed_limit
        map_data['map_polygon']['stop_sign_mask'] = polygon_stop_sign_mask
        map_data['map_polygon']['stop_sign_position'] = polygon_stop_sign_position
        if self.traffic_signal:
            map_data['map_polygon'][
                'trafic_signal_mask'] = polygon_trafic_signal_mask[:, :self.num_historical_steps].clone().detach()
            map_data['map_polygon'][
                'traffic_signal_state'] = polygon_traffic_signal_state[:, :self.num_historical_steps].clone().detach()
            map_data['map_polygon']['stop_point'] = polygon_stop_point[:, :self.num_historical_steps].clone().detach()
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    def _download(self) -> None:
        # if complete raw/processed files exist, skip downloading
        if ((os.path.isdir(self.raw_dir) and len(self.raw_file_names) == self._num_raw_files) or
                (os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self))):
            return
        if os.path.isdir(os.path.join(self.root, self.dir)):
            self._raw_file_names = [name for name in os.listdir(os.path.join(self.root, self.dir)) if
                                    os.path.isfile(os.path.join(self.root, self.dir, name))]
            if len(self.raw_file_names) == self._num_raw_files:
                if os.path.isdir(self.raw_dir):
                    shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                for raw_file_name in self.raw_file_names:
                    shutil.move(os.path.join(self.root, self.dir, raw_file_name), self.raw_dir)
                return
            else:
                shutil.rmtree(os.path.join(self.root, self.dir))
                self._raw_file_names = []
        self._processed_file_names = []
        self.download()

    def _process(self) -> None:
        # if complete processed files exist, skip processing
        if os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self):
            return
        print('Processing...', file=sys.stderr)
        if os.path.isdir(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                if name.endswith(('pkl', 'pickle')):
                    os.remove(os.path.join(self.processed_dir, name))
        else:
            os.makedirs(self.processed_dir)
        self._processed_file_names = []
        self.process()
        print('Done!', file=sys.stderr)
