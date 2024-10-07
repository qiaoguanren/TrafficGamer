# Visualization : 3D and topdown visualization of scenarios

## Getting Started : simulation with ScenarioNet

**Step 1**: Establish ScenarioNet environment
The detailed installation guidance is available at [documentation](https://scenarionet.readthedocs.io/en/latest/)

```
# create environment
conda create -n scenarionet python=3.9
conda activate scenarionet

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e.

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

**Step 2**: Build Scenarionet dataset
we need to install and parse the original data and convert to scenario description of Scenarionet.

For original data of argoverse2 and waymo, you can follow [instructions](https://scenarionet.readthedocs.io/en/latest/datasets.html)
```
pip install tensorflow==2.12.0
conda install protobuf==3.20

# prepare data and build dataset
# For Argoverse2
python -m scenarionet.convert_argoverse2 -d /path/to/exp_converted/ --raw_data_path /path/to/exp_av2

# For Waymo
python -m scenarionet.convert_waymo -d /path/to/exp_converted/ --raw_data_path /path/to/exp_waymo
```

**Step 3**: Simulation and Visualization
The database can be loaded to MetaDrive simulator for scenario replay or closed-loop simulation.
```
python -m scenarionet.sim -d /path/to/exp_converted --render 2D
```
By adding --render 3D flag, we can use 3D renderer:
```
python -m scenarionet.sim -d /path/to/exp_converted --render 3D
```

Besides, if you want to have other need of visualization, such as changing the focal vehicle, interesting vehicles, and producing a lots of visualization with same scenario and different tracking data quickly, you can use the py files we provide.
```
python waymo_vis.py -d /path/to/data_path --scenario_id exp_id --pkl_path exp_pkl
python av2_vis.py -d /path/to/data_path --scenario_id exp_id --pkl_path exp_pkl
```

For example, when you use waymo_vis.py, your directory might look like the following.
scenario
├── sd_waymo_v1.2_9c6eb32bcc69d42e_scenario1_distance_limit8.0_penalty_initial_value1.0_cost_quantile8.pkl
├── sd_waymo_v1.2_9c6eb32bcc69d42e_scenario1_distance_limit8.0_penalty_initial_value1.0_cost_quantile56.pkl
└── ...
waymo_dataset
├── sd_waymo_v1.2_9c6eb32bcc69d42e.pkl
├── dataset_mapping.pkl
└── dataset_summary.pkl
data_path = /path/to/waymo_dataset, scenario_id = 9c6eb32bcc69d42e, pkl_path=/path/to/scenario

or use av2_vis.py
scenario
├── d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84_distance_limit2.0_penalty_initial_value1.0_cost_quantile8_seed123
|   ├── scenario_d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84_2024_07_24_08_59_18.parquet
├── d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84_distance_limit2.0_penalty_initial_value1.0_cost_quantile8_seed666
|   ├── scenario_d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84_2024_07_24_09_01_08.parquet
└── ...
av2_dataset
├── sd_waymo_v1.2_9c6eb32bcc69d42e.pkl
├── dataset_mapping.pkl
└── dataset_summary.pkl
data_path = /path/to/av2_dataset, scenario_id = d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84, pkl_path=/path/to/scenario



 

