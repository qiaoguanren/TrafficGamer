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
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario import utils as sd_utils
def make_GIF(frames, name="demo.gif"):
    # print("Generate gif...")
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    print(f"GIF saved as {name}")
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

def read_files_starting_with(data_path, directory, start_with, video_length):
    for filename in os.listdir(directory):
        if filename.startswith(start_with):
            file_path = os.path.join(directory, filename)
            df2 = pd.read_pickle(file_path)
            map_file = f'{data_path}/dataset_mapping.pkl'
            sum_file = f'{data_path}/dataset_summary.pkl'
            
            storagename = f'{data_path}/{start_with}.pkl'
            remove_list = []
            tracks_list = []
            for key, value in df2['tracks'].items():
                if not np.all(value['state']['valid'][:11]==1):
                    tracks_list.append(key)
            for key,value in df2['map_features'].items():
                if value['type']=='LANE_UNKNOWN':
                    remove_list.append(key)
            for i in remove_list:
                df2['map_features'].pop(i)
            for i in tracks_list:
                df2['tracks'].pop(i)
            if objects_of_interest!=[]:
                df2['metadata']['objects_of_interest'] = objects_of_interest
            if sdc_id!='-1':
                df2['metadata']['sdc_id'] = sdc_id
            map_dict = {f'{start_with}.pkl':''}
            sum_dict = {f'{start_with}.pkl':df2['metadata']}
            with open(storagename, 'wb') as f:
                pickle.dump(df2, f)
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
    parser.add_argument("--video_length", default=8, help="Specifying the video length")
    parser.add_argument("--dataset_version", default=1.2, help="Specifying the version of waymo dataset")
    parser.add_argument("--objects_of_interest", default=[], help="Specifying the objects of interests that will be marked with orangle color")
    parser.add_argument("--sdc_id", default='-1', help="Specifying the focal object")
    args = parser.parse_args()
    start_with_value = f'sd_waymo_v{args.dataset_version}_{args.scenario_id}'
    read_files_starting_with(args.database_path ,args.pkl_path, start_with_value, args.video_length, args.objects_of_interest, args.sdc_id)






