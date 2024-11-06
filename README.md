# TrafficGamer

This repository is the official implementation of the TrafficGamer.
 [[paper]](https://arxiv.org/abs/2408.15538) [[demo]](https://qiaoguanren.github.io/trafficgamer-demo/)

## Getting Started

**Step 1**: clone this repository:

```
git clone https://github.com/qiaoguanren/TrafficGamer.git && cd TrafficGamer
```

**Step 2**: create a conda environment and install the dependencies:
```
conda create -n TrafficGamer python=3.8
conda activate TrafficGamer
pip install -r requirements.txt
```

**Step 3**: install the [Argoverse 2 API](https://github.com/argoverse/av2-api) and download the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html) following the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html). Install the [Waymo API](https://github.com/waymo-research/waymo-open-dataset) and download the [Waymo Motion Forecasting Dataset v1.2.0](https://waymo.com/open/download/motion-forecasting/) following the [Waymo User Guide](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/user_guide.md).

## Pre-Training & Fine-tuning

### Pre-Training
For pre-training, we require all the training data from the entire dataset.
You can train the auto-regressive pre-train model on 8 NVIDIA GeForce RTX 4090:
```
python train_qcnet.py --root /path/to/dataset_root/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150
```
or 
```
python train_waymo.py --root /path/to/dataset_root/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset waymo --num_historical_steps 11 --num_future_steps 80 --num_recurrent_steps 4 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150
```

**Note 1**: when running the training script for the first time, it will take several hours to preprocess the data.

**Note 2**: during training, the checkpoints will be saved in `lightning_logs/` automatically. 

**Note 3**: you can adjust the batch size and the number of devices.

### Validation

To evaluate on the validation set:
```
python val.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```

### Fine-tuning
The data we used for fine-tuning is selected from the validation set of the two datasets, consisting of six distinct scenes, each with unique characteristics. The data is all collected from open-source real-world data obtained through sensors. The data has been preprocessed by our code and saved in a binary .pkl file using pickle. It includes various traffic features such as vehicle positions, speeds, times, and cities, as well as map information like lane markings and intersections.
To generate more safety-critical scenarios, you should follow these steps:

**Step 1**: Select an interesting scene from the data and record the id of scenario.

**Step 2**: Select the controlled vehicles and determine their respective destinations.

**Step 3**: Load the checkpoint of the pre-trained model and Load the data with DataLoader.

**Step 4**: To estimate CCE, you can choose different algorithms and run the fine-tuning script:
```
python run_parallel.py --filename "train_trafficgamer.py --magnet --track" --seed 123 --scenario 1 --workspace CCE-MAPPO
```

**Step 5**: To generate all kinds of scenarios, you can run the fine-tuning script:
```
python run_parallel.py --filename "train_trafficgamer.py --magnet --track" --seed 123 --scenario 2 --workspace TrafficGamer --distance_limit 3.0 --cost_quantile 32
```
To get better performance, you can try to adjust the `--eta-coef1`, `--penalty_initial_value`, `--distance_limit` and `--cost_quantile` parameters.

**Step 6**: We will record the reward. Then we will fix the other agents' RL parameters and use the script:
```
python optimal_value.py
```
to get the optimal value of the chosen agent by utilizing the PPO algorithm. We take the difference between the optimal value and real rewards. This is the first calculation of estimating CCE.

**Step 7**: To use the second method for calculating CCE, you need to set the parameter confined_action=True and retrain the agent:
```
python run_parallel.py --filename "train_trafficgamer.py --magnet --track --confined_action" --seed 123 --scenario 1 --workspace CCE-MAPPO
```
At the same time, We get expert rewards by picking out the best strategy from the limited strategy set.
```
python expert_policy.py
```
We take the difference between expert rewards and real rewards. This is the second calculation of estimating CCE.

**Step 8**: You can check your 2D visualization in your wandb project or you can save .mp4 video in a folder and run:
```
python serve.py
```

**Step 9**: You can use the 3D visualization method following the readme.md file in the visualization folder.

## License

This repository is licensed under [BSD 3-Clause License](LICENSE).

## Citation

If you use the code of this repo or find it is helpful for your research, please cite:
```
@article{qiao2024trafficgamer,
  title={TrafficGamer: Reliable and Flexible Traffic Simulation for Safety-Critical Scenarios with Game-Theoretic Oracles},
  author={Qiao, Guanren and Quan, Guorui and Yu, Jiawei and Jia, Shujun and Liu, Guiliang},
  journal={arXiv preprint arXiv:2408.15538},
  year={2024}
}
```

## Acknowledgement

This work is based on the code repo: [QCNet](https://github.com/ZikangZhou/QCNet). We are grateful that this project has provided a solid foundation for our ideas. 
