import pandas as pd
import pickle
import numpy as np
import math
import os
import mediapy
import argparse
from datetime import datetime
from IPython.display import Image as IImage
import pygame
import numpy as np
from PIL import Image
from metadrive.engine.engine_utils import close_engine
close_engine()
from metadrive.pull_asset import pull_asset
pull_asset(False)
# NOTE: usually you don't need the above lines. It is only for avoiding a potential bug when running on colab
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario import utils as sd_utils

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
    
    if current_length > max_length:
        max_length = current_length
    
    return max_length
def read_files_starting_with(data_path, directory, scenario_id, video_length,objects_of_interest=[], sdc_id='-1'):
    for filename in os.listdir(directory):
        if filename.startswith(scenario_id):
            dir_path = os.path.join(directory, filename)
            for filename2 in os.listdir(dir_path):
                if filename2.startswith(f'scenario_{scenario_id}'):
                    file_path = os.path.join(dir_path, filename2)
                    df2 = pd.read_parquet(file_path)
                    df_o = pd.read_pickle(f"{data_path}/sd_av2_v2_{scenario_id}.pkl")
                    # df2['metadata']['sdc_id'] = '758'
                    # df2['metadata']['sdc_id'] = '1007'
                    map_file = f'{data_path}/dataset_mapping.pkl'
                    sum_file = f'{data_path}/dataset_summary.pkl'
                    
                    storagename = f'{data_path}/sd_av2_v2_{scenario_id}.pkl'

                    for key in df_o['tracks'].keys():
                        pos = df_o['tracks'][key]['state']['position']
                        heading = df_o['tracks'][key]['state']['heading']
                        vel = df_o['tracks'][key]['state']['velocity']
                        valid = df_o['tracks'][key]['state']['valid']
                        total_distance = 0
                        pos_shape = pos.shape
                        key_cor = df2[df2['track_id']==key]
                        if df2[df2['track_id']==key]['observed'][:50].sum()!=50:
                            valid = np.zeros(110,)
                        else:
                            valid = np.ones(110,)
                        for i in range(pos_shape[0]):
                            state_cor = key_cor[key_cor['timestep']==i]
                            if not state_cor.empty:
                                pos[i,0]=state_cor['position_x'].iloc[0]
                                pos[i,1] = state_cor['position_y'].iloc[0]
                                heading[i]=state_cor['heading'].iloc[0]
                                vel[i,0]=state_cor['velocity_x'].iloc[0]
                                vel[i,1]=state_cor['velocity_y'].iloc[0]
        
                                if (i>=1) and (np.any(pos[i-1,:2]!=0)):
                                    x1, y1 = pos[i - 1,:2]
                                    x2, y2 = pos[i,:2]
                                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    total_distance += distance
    
                        df_o['metadata']['object_summary'][i]={'type':'VEHICLE','object_id':i,'track_length':110, 'moving_distance':total_distance, 'valid_length':valid.sum(), 'continuous_valid_length':continuous_valid_length(valid)}
                        df_o['tracks'][key]['state']['length'] *= valid
                        df_o['tracks'][key]['state']['width'] *= valid
                        df_o['tracks'][key]['state']['height'] *= valid
                        df_o['tracks'][key]['state']['position'] = pos
                        df_o['tracks'][key]['state']['heading'] = heading
                        df_o['tracks'][key]['state']['velocity'] = vel
                        df_o['tracks'][key]['state']['valid'] = valid

                    if len(set(df2['track_id']) - set(df_o['tracks'].keys()))>0:
                        diff = set(df2['track_id']) - set(df_o['tracks'].keys())
                        list_df2 = list(df2['track_id'])
                        print(diff)
                        df_o['metadata']['objects_of_interest'] += diff
                        df_o['metadata']['number_summary']['num_objects'] += len(diff)
                        df_o['metadata']['number_summary']['num_objects_each_type']['VEHICLE'] += len(diff)
                        df_o['metadata']['number_summary']['num_moving_objects']+= len(diff)
                        df_o['metadata']['number_summary']['num_moving_objects_each_type']['VEHICLE']+= len(diff)
                        for i in list(diff):
                            key_cor = df2[df2['track_id']==i]
                            n = list_df2.count(i)
                            pos = np.zeros((n,3))
                            heading = np.zeros((n,))
                            vel = np.zeros((n,2))
                            length = np.zeros((n,))
                            width = np.zeros((n,))
                            height = np.zeros((n,))
                            valid = np.zeros((n,))
                            total_distance = 0.0

                            for j in range(n):
                                state_cor = key_cor[key_cor['timestep']==j]
                                if not state_cor.empty:
                                    pos[j,0]=state_cor['position_x'].iloc[0]
                                    pos[j,1] = state_cor['position_y'].iloc[0]
                                    heading[j]=state_cor['heading'].iloc[0]
                                    vel[j,0]=state_cor['velocity_x'].iloc[0]
                                    vel[j,1]=state_cor['velocity_y'].iloc[0]
                                    valid[j]=state_cor['observed'].iloc[0]
                                    length[j] = 4.0*valid[j]
                                    width[j]=2.0*valid[j]
                                    height[j] = 1.0*valid[j]
                                    if (j>=1) and (np.any(pos[j-1,:2]!=0)):
                                        x1, y1 = pos[j - 1,:2]
                                        x2, y2 = pos[j,:2]
                                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                        total_distance += distance
                            df_o['metadata']['object_summary'][i]={'type':'VEHICLE','object_id':i,'track_length':110, 'moving_distance':total_distance, 'valid_length':valid.sum(), 'continuous_valid_length':continuous_valid_length(valid)}
                            tracklet = {}
                            tracklet['type'] = 'VEHICLE'
                            tracklet['metadata'] = {'track_length': df2[df2['track_id']==i]['num_timestamps'].iloc[0], 'type': 'VEHICLE', 'object_id': i, 'dataset': 'av2'}
                            tracklet['state'] = {'position':pos, 'length':length, 'width':width, 'height':height, 'heading':heading, 'velocity':vel, 'valid':valid}
                            df_o['tracks'][i]=tracklet
                    if objects_of_interest!=[]:
                        df_o['metadata']['objects_of_interest'] = objects_of_interest
                    if sdc_id!='-1':
                        df_o['metadata']['sdc_id'] = sdc_id
                    # df_o['metadata']['objects_of_interest'].append('100557')
                    # df_o['metadata']['objects_of_interest'].append('22588')
                    # df_o['metadata']['objects_of_interest'] = ['253391', '248391', '251391', '252391', '250391', '249391','247268']
                    # df_o['metadata']['objects_of_interest'].append('11830')
                    # df_o['metadata']['objects_of_interest'] = ['181283', '180283', '184283', '183283', '182283', '179283','177020']
                    # df_o['metadata']['objects_of_interest'] = ['11015', '9015', '8015', '10015','6967']
                    # df_o['metadata']['sdc_id'] = '104267'
                    # df_o['metadata']['sdc_id'] = '27620'
                    # df_o['metadata']['sdc_id'] = '249391'
                    # df_o['metadata']['sdc_id'] = '14847'
                    # df_o['metadata']['sdc_id'] = '183283'
                    # df_o['metadata']['sdc_id'] = '11015'
                    map_dict = {f'sd_av2_v2_{scenario_id}.pkl':''}
                    sum_dict = {f'sd_av2_v2_{scenario_id}.pkl':df_o['metadata']}
                    with open(storagename, 'wb') as f:
                        pickle.dump(df_o, f)
                    with open(map_file, 'wb') as f:
                        pickle.dump(map_dict, f)
                    with open(sum_file, 'wb') as f:
                        pickle.dump(sum_dict, f)
                    threeD_render=True # turn on this to enable 3D render. It only works when you have a screen and not running on Colab.
                    os.environ["SDL_VIDEODRIVER"] = "dummy" # Hide the pygame window

                    env = ScenarioEnv(
                        {
                            "manual_control": False,
                            "show_interface": True,
                            "show_logo": True,
                            "show_fps": True,
                            "use_render": threeD_render,
                            "force_reuse_object_name":True,
                            "agent_policy": ReplayEgoCarPolicy,
                            "data_directory": data_path,
                            "num_scenarios": 1, 
                        }
                    )
                    
                    generate_video = True

                    folder_name = "{0}_video_{1}".format(filename,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                    video_bev = []
                    video_interface = []


                    ep_count = 0


                    for seed in range(1): 
                        print("\nSimulate Scenario: {}".format(seed))
                        env.reset(seed=seed)
                        frames = []
                        for i in range(1, 10000):
                            o, r, tm, tc, info = env.step([0, 0.])
                            img_interface = env.main_camera.perceive(to_float=False)
                            img_interface = img_interface[..., ::-1]
                            img_bev = env.render(
                                mode="topdown",
                                target_vehicle_heading_up=False,
                                draw_target_vehicle_trajectory=True,
                                film_size=(3000, 3000),
                                screen_size=(800, 800),
                            )

                            if generate_video:
                                img_bev = img_bev.swapaxes(0, 1)
                                video_bev.append(img_bev)
                                video_interface.append(img_interface)


                            if tm or tc:
                                ep_count += 1

                                env.engine.force_fps.disable()
                                if generate_video:
                                    os.makedirs(folder_name, exist_ok=True)
                                    fps = len(video_bev)/video_length

                                    video_base_name = "{}/{}".format(folder_name, ep_count)
                                    video_name_bev = video_base_name + "_bev.mp4"
                                    print("BEV video should be saved at: ", video_name_bev)
                                    mediapy.write_video(video_name_bev, video_bev, fps=fps)

                                    video_name_interface = video_base_name + "_interface.mp4"
                                    print("3D Interface video should be saved at: ", video_name_interface)
                                    mediapy.write_video(video_name_interface, video_interface, fps=fps)
                                break
                    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", "-d", required=True, help="The path of the database")
    parser.add_argument("--scenario_id", required=True, help="Specifying a scenario to run")
    parser.add_argument("--pkl_path", required=True, help="Specifying the corresponding scenario pkl file")
    parser.add_argument("--video_length", default=6, help="Specifying the video length")
    parser.add_argument("--objects_of_interest", default=[], help="Specifying the objects of interests that will be marked with orangle color")
    parser.add_argument("--sdc_id", default='-1', help="Specifying the focal object")
    args = parser.parse_args()
    read_files_starting_with(args.database_path ,args.pkl_path, args.scenario_id, args.video_length, args.objects_of_interest, args.sdc_id)







