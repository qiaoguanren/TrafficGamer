# TrafficGamer

This repository is the official implementation of the TrafficGamer.

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

To generate more safety-critical scenarios, you should follow these steps:
**Step 1**: Select an interesting scene from the dataset and choose the processed data format.

**Step 2**: Select the controlled vehicles and determine their respective destinations.

**Step 3**: To estimate CCE, you can run the fine-tuning script:
```
python run_parallel.py --filename "train_trafficgamer.py --magnet --track" --seed 123 --scenario 1 --workspace CCE-MAPPO
```

**Step 4**: To generate all kinds of scenarios, you can run the fine-tuning script:
```
python run_parallel.py --filename "train_trafficgamer.py --magnet --track" --seed 123 --scenario 2 --workspace TrafficGamer --distance_limit 3.0 --cost_quantile 32
```

**Step 5**: You can check your 2D visualization in your wandb project or you can save .mp4 video in a folder and run:
```
python serve.py
```

**Step 6**: You can use the 3D visualization method following the readme.md file in visualization folder.

## License

This repository is licensed under [BSD 3-Clause License](LICENSE).

## Acknowledgement

This work is based on the code repo: [QCNet](https://github.com/ZikangZhou/QCNet).  