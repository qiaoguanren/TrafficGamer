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

import io
import math
import pandas as pd
from pathlib import Path

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image as img
from typing import Final
from random import choices
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from datetime import datetime
from utils.utils import get_auto_pred, get_transform_mat
from shapely.geometry import LineString

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = (
    100  # Ensure actor bounding boxes are plotted on top of all map elements
)

_STATIC_OBJECT_TYPES = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}
_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7


from matplotlib.patches import Rectangle
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)


def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_static_map_elements_waymo(
    static_map, show_ped_xings: bool = False
) -> None:

    for key, value in static_map.items():
        if value['type']=='DRIVEWAY':
            _plot_polygon(value['polygon'], alpha=0.5, color='grey')
    # Plot lane segments
        elif value['type']=='ROAD_LINE_SOLID_SINGLE_YELLOW':
            _plot_polyline(
                value['polyline'],
                style='-',
                line_width=0.5,
                color='orange',
            )
        elif value['type']=='ROAD_LINE_SOLID_SINGLE_WHITE':
            _plot_polyline(
                value['polyline'],
                style='-',
                line_width=0.5,
                color='grey',
            )
        elif value['type']=='ROAD_LINE_BROKEN_SINGLE_WHITE':
            _plot_polyline(
                value['polyline'],
                style='--',
                line_width=0.5,
                color='grey',
            )
        elif value['type']=='ROAD_LINE_BROKEN_SINGLE_YELLOW':
            _plot_polyline(
                value['polyline'],
                style='--',
                line_width=0.5,
                color='orange',
            )
        
        elif value['type']=='ROAD_EDGE_BOUNDARY':
            _plot_polyline(
                value['polyline'],
                style='-',
                line_width=0.5,
                color='black',
            )
        elif value['type']=='ROAD_LINE_SOLID_DOUBLE_YELLOW':
            _plot_polyline(
                value['polyline'],
                style='-',
                line_width=0.5,
                color='orange',
            )
            ano_line = value['polyline'].copy()
            ano_line[:,0] += 0.5
            ano_line[:,1] += 0.5
            _plot_polyline(
                ano_line,
                style='-',
                line_width=0.5,
                color='orange',
            )
        elif value['type']=='LANE_FREEWAY':
            _plot_polyline(
                value['polyline'],
                style='--',
                line_width=0.5,
                color='green',
            )
        elif value['type']=='LANE_SURFACE_STREET':
            _plot_polyline(
                value['polyline'],
                style='--',
                line_width=0.5,
                color='green',
            )

def _plot_actor_tracks(ax: plt.Axes, scenario, timestep: int):
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        # Plot polyline for focal agent location history
        track_color = _DEFAULT_ACTOR_COLOR
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines(
                [
                    np.array(
                        [
                            list(object_state.position)
                            for object_state in track.object_states
                            if object_state.timestep > timestep
                        ]
                    )
                ],
                color="black",
                line_width=2,
                style="--",
            )
            _plot_polylines([actor_trajectory], color=track_color, line_width=2)
        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds


def _plot_polylines(
    polylines,
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )

def _plot_polyline(
    polyline,
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    plt.plot(
        polyline[:, 0],
        polyline[:, 1],
        style,
        linewidth=line_width,
        color=color,
        alpha=alpha,
        )

def _plot_polygons(polygons, *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)
def _plot_polygon(polygon, *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location,
    heading: float,
    color: str,
    bbox_size,
    annotation_text=None,
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
        annotation_text: Text to be displayed inside the bounding box. If None, no text will be displayed.
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        np.degrees(heading),
        color=color,
        zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)

    #Add annotation text inside the rectangle if annotation_text is not None
    # if annotation_text is not None:
    #     ax.text(
    #         pivot_x,
    #         pivot_y,
    #         annotation_text,
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #         fontsize=10,
    #         color="black",
    #         weight="bold",
    #         zorder=_BOUNDING_BOX_ZORDER + 1,
    #     )


def plot_traj(scenario_static_map, scenario, bounds=80.0):
    plot_bounds = (0, 0, 0, 0)

    _, ax = plt.subplots()

    _plot_static_map_elements(scenario_static_map)
    cur_plot_bounds = _plot_actor_tracks(ax, scenario, 50)
    if cur_plot_bounds:
        plot_bounds = cur_plot_bounds
    plt.xlim(
        plot_bounds[0] - bounds,
        plot_bounds[1] + bounds,
    )
    plt.ylim(
        plot_bounds[2] - bounds,
        plot_bounds[3] + bounds,
    )
    # plt.scatter(702, -925, c='r', marker='x')
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


def _plot_actor_tracks_with_data(ax: plt.Axes, data, agent_number, timestep: int):
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for i in range(data["agent"]["num_nodes"]):
        if not (data["agent"]["valid_mask"][i, timestep]):
            continue
        # Get timesteps for which actor data is valid
        track_color = _DEFAULT_ACTOR_COLOR

        # Get actor trajectory and heading history

        actor_trajectory = data["agent"]["position"][i, : timestep + 1][
            data["agent"]["valid_mask"][i, : timestep + 1]
        ].cpu()
        actor_headings = data["agent"]["heading"][i, : timestep + 1][
            data["agent"]["valid_mask"][i, : timestep + 1]
        ].cpu()

        if data["agent"]["category"][i] == TrackCategory.FOCAL_TRACK.value:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR

            _plot_polylines([actor_trajectory], color="black", line_width=2)
            ax.arrow(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                data["agent"]["velocity"][i, timestep, 0].cpu(),
                data["agent"]["velocity"][i, timestep, 1].cpu(),
                width=0.1,
                length_includes_head=True,
                head_width=0.75,
                head_length=1,
                fc="r",
                ec="b",
            )
        elif i == data["agent"]["av_index"]:
            track_color = _FOCAL_AGENT_COLOR
        # elif i==data['agent']['num_nodes']-agent_number and i!=data['agent']['num_nodes']-1:
        elif data["agent"]["category"][i] == TrackCategory.SCORED_TRACK.value:
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color="black", line_width=2)
            ax.arrow(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                data["agent"]["velocity"][i, timestep, 0].cpu(),
                data["agent"]["velocity"][i, timestep, 1].cpu(),
                width=0.1,
                length_includes_head=True,
                head_width=0.75,
                head_length=1,
                fc="r",
                ec="b",
            )
            agent_number -= 1
        elif data["agent"]["type"][i] > 4:
            continue

        # Plot bounding boxes for all vehicles and cyclists
        if data["agent"]["type"][i] == 0:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
                str(i) if data["agent"]["category"][i]>=TrackCategory.SCORED_TRACK.value else None,
            )
        elif data["agent"]["type"][i] == 2 or data["agent"]["type"][i] == 3:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds
def _plot_actor_tracks_with_data_waymo(ax: plt.Axes, data, agent_number, timestep: int, choose_agent):
    """Plot all actor tracks (up to a particular time step) associated with an Waymo scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for i in range(data["agent"]["num_nodes"]):
        if not (data["agent"]["valid_mask"][i, timestep]):
            continue
        # Get timesteps for which actor data is valid
        track_color = _DEFAULT_ACTOR_COLOR

        # Get actor trajectory and heading history
        actor_trajectory = data["agent"]["position"][i, : timestep + 1][
            data["agent"]["valid_mask"][i, : timestep + 1]
        ].cpu()
        actor_headings = data["agent"]["heading"][i, : timestep + 1][
            data["agent"]["valid_mask"][i, : timestep + 1]
        ].cpu()
        
        if i in choose_agent and track_bounds==None:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR

            _plot_polylines([actor_trajectory], color="black", line_width=2)
            ax.arrow(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                data["agent"]["velocity"][i, timestep, 0].cpu(),
                data["agent"]["velocity"][i, timestep, 1].cpu(),
                width=0.1,
                length_includes_head=True,
                head_width=0.75,
                head_length=1,
                fc="r",
                ec="b",
            )
        elif i == data["agent"]["av_index"]:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR

            _plot_polylines([actor_trajectory], color="black", line_width=2)
            ax.arrow(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                data["agent"]["velocity"][i, timestep, 0].cpu(),
                data["agent"]["velocity"][i, timestep, 1].cpu(),
                width=0.1,
                length_includes_head=True,
                head_width=0.75,
                head_length=1,
                fc="r",
                ec="b",
            )
        #elif i==data['agent']['num_nodes']-agent_number and i!=data['agent']['num_nodes']-1:
        elif i in choose_agent and track_bounds!=None:
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color="black", line_width=2)
            ax.arrow(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                data["agent"]["velocity"][i, timestep, 0].cpu(),
                data["agent"]["velocity"][i, timestep, 1].cpu(),
                width=0.1,
                length_includes_head=True,
                head_width=0.75,
                head_length=1,
                fc="r",
                ec="b",
            )
            agent_number -= 1

        # Plot bounding boxes for all vehicles and cyclists
        if data["agent"]["type"][i] == 1:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
                str(i) if i in choose_agent else None,
            )
        elif data["agent"]["type"][i] == 2 or data["agent"]["type"][i] == 3:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0],
                actor_trajectory[-1, 1],
                "o",
                color=track_color,
                markersize=4,
            )

    return track_bounds

def plot_traj_with_data(data, scenario_static_map, agent_number, scenario_num,t=50, bounds=80.0, dataset_type='av2', choose_agent=None):
    plot_bounds = (0, 0, 0, 0)

    _, ax = plt.subplots()
    if dataset_type=='av2':
        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks_with_data(ax, data, agent_number, t)
    elif dataset_type=='waymo':
        _plot_static_map_elements_waymo(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks_with_data_waymo(ax, data, agent_number, t,choose_agent=choose_agent)
        
    
    if cur_plot_bounds:
        plot_bounds = cur_plot_bounds
    plt.xlim(
        # 5256 - bounds,
        # 5265 + bounds,
        # 2640 - bounds,
        # 2670 + bounds,
        # 690 - bounds,
        # 695 + bounds,
        # -8339.5 - bounds,
        # -8339.5 + bounds,

        plot_bounds[0] - bounds,
        plot_bounds[1] + bounds,
        #-4790, -4688#3
        #828.5, 928.54 #2
        #-9340, -9240 #6
        #-4723, -4623 #4
        #3440, 3541#5
        #-1401, -1297#1
    )
    #print(plot_bounds[0] - bounds, plot_bounds[1] + bounds)
    plt.ylim(
        # 295 - bounds,
        # 330 + bounds,
        # -2365 - bounds,
        # -2360 + bounds,
        # -911 - bounds,
        # -900 + bounds,
        # -840 - bounds,
        # -830 + bounds,

        #-3993.6, -3893.44 #2
        #12938, 13038 #6
        #4047, 4148 #4
        #-6252, -6151#3
        #-168, -55#5
        #3822, 3923#1
        plot_bounds[2] - bounds,
        plot_bounds[3] + bounds,
    )
    #print(plot_bounds[2] - bounds, plot_bounds[3] + bounds)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("/home/guanren/Multi-agent-competitive-environment/results/scenario1_raw.png", bbox_inches="tight", pad_inches=0)


def generate_video(new_input_data, scenario_static_map, model, vid_path):
    frames = []
    with torch.no_grad():
        offset = 5
        new_data = new_input_data.cuda().clone()
        pred = model(new_data)
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
        origin, theta, rot_mat = get_transform_mat(new_data, model)
        new_true_trans_position_refine = torch.einsum(
            "bijk,bkn->bijn",
            pred["loc_refine_pos"][..., : model.output_dim],
            rot_mat.swapaxes(-1, -2),
        ) + origin[:, :2].unsqueeze(1).unsqueeze(1)
        auto_pred = pred
        init_origin, init_theta, init_rot_mat = get_transform_mat(new_data, model)
        agent_index = torch.nonzero(
            new_input_data["agent"]["category"] == 3, as_tuple=False
        ).item()
        pi = pred["pi"]
        eval_mask = new_data["agent"]["category"] == 3
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        for i in range(0, 110):
            if i < 50:
                plot_traj_with_data(new_data, scenario_static_map, 8, bounds=60, t=i)
            else:
                if (110 - i) % offset == 0:
                    # true_trans_position_propose=new_true_trans_position_propose
                    true_trans_position_refine = new_true_trans_position_refine
                    print(
                        torch.sqrt(
                            traj_propose[
                                new_data["agent"]["category"] == 3, :, offset, 0
                            ]
                            ** 2
                            + traj_refine[
                                new_data["agent"]["category"] == 3, :, offset, 1
                            ]
                            ** 2
                        )
                    )
                    # max_value = -1
                    # max_index = -1
                    # for k in range(6):
                    #     x = traj_propose[new_data["agent"]["category"] == 3, k, offset, 0].cpu()
                    #     y = traj_propose[new_data["agent"]["category"] == 3, k, offset, 1].cpu()
                    #     value = np.sqrt(x**2 + y**2)
                    #     if value > max_value:
                    #         max_value = value
                    #         max_index = k

                    #   print(reward_function(new_input_data,new_data,model,agent_index))
                    # reg_mask = new_data['agent']['predict_mask'][:, model.num_historical_steps:]
                    # cls_mask = new_data['agent']['predict_mask'][:, -1]
                    # gt = torch.cat([data['agent']['target'][...,timestep:timestep+offset, :model.output_dim], data['agent']['target'][...,timestep:timestep+offset, -1:]], dim=-1)
                    # l2_norm = (torch.norm(traj_propose[:-1,...,:offset, :model.output_dim] -
                    #                     gt[..., :model.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask[:-1,...,:offset].unsqueeze(1)).sum(dim=-1)
                    # best_mode = l2_norm.argmin(dim=-1)
                    # best_mode=torch.randint(6,size=(new_data['agent']['num_nodes'],))
                    best_mode = pi_eval.argmax(dim=-1)
                    # best_mode[new_data["agent"]["category"] == 3] = max_index
                    # best_mode[-1] = 4
                    # best_mode[agent_index]=5
                    (
                        new_data,
                        auto_pred,
                        _,
                        _,
                        (
                            new_true_trans_position_propose,
                            new_true_trans_position_refine,
                        ),
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
                    pi_eval = F.softmax(auto_pred["pi"][eval_mask], dim=-1)
                    plot_traj_with_data(
                        new_data, scenario_static_map, 8, bounds=60, t=50 - offset
                    )
                    # for j in range(6):
                    #     xy = true_trans_position_refine[
                    #         new_data["agent"]["category"] == 3
                    #     ][0].cpu()
                    #     plt.plot(xy[j, ..., 0], xy[j, ..., 1])
                else:
                    plot_traj_with_data(
                        new_data,
                        scenario_static_map,
                        8,
                        bounds=60,
                        t=50 - offset + (i - 50) % offset,
                    )
                    # for j in range(6):
                    #     xy = true_trans_position_refine[
                    #         new_data["agent"]["category"] == 3
                    #     ][0].cpu()
                    #     plt.plot(
                    #         xy[j, (i - 50) % offset :, 0], xy[j, (i - 50) % offset :, 1]
                    #     )

            plt.title(f"timestep={i}")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            frame = img.open(buf)
            frames.append(frame)
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()
    
def plot_destination(scenario):
    if scenario==2:
        plt.scatter(2691, -2361,c='r',marker='x')
        plt.scatter(2680, -2380,c='r',marker='x')
        plt.scatter(2677, -2363,c='r',marker='x')
    elif scenario==4:
        plt.scatter(-8340, -813, c='r', marker='x')
        plt.scatter(-8353, -825, c='r', marker='x')
        plt.scatter(-8327, -827, c='r', marker='x')
        plt.scatter(-8342, -837, c='r', marker='x')
    elif scenario==3:
        plt.scatter(5043, 3019, c='r', marker='x')
    elif scenario==1:
        plt.scatter(5193,145,c='r',marker='x')
        plt.scatter(5197,130,c='r',marker='x')
        plt.scatter(5181,133,c='r',marker='x')
    elif scenario==5:
        plt.scatter(5749,3293,c='r',marker='x')
    elif scenario==6:
        plt.scatter(3545, 435,c='r',marker='x')
        plt.scatter(3590, 450,c='r',marker='x')
    