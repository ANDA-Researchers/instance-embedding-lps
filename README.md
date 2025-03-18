# ANDA-Vision Team

## Introduction

This is an official implementation of our TIV's paper ["Instance Embedding as Queries for DETR-based LiDAR Panoptic Segmentation"](https://doi.org/10.1109/TIV.2024.3488035).

If you find the code useful for your research, please consider citing our paper:
```
@ARTICLE{10738293,
  author={Ha-Phan, Ngoc-Quan and Vuong, Hung Viet and Yoo, Myungsik},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Instance Embedding as Queries for DETR-based LiDAR Panoptic Segmentation}, 
  year={2024},
  pages={1-13},
  doi={10.1109/TIV.2024.3488035}}

```

## Installation

```
sudo chmod +x env.sh
bash env.sh
```

## Usage

### Data preparation

#### Semantickitti

```text
data/
├── semantickitti
│   ├── sequences
│   │   ├── 00
│   │   |   ├── labels
│   │   |   ├── velodyne
│   │   ├── 01
│   │   ├── ...
│   ├── semantickitti_infos_train.pkl
│   ├── semantickitti_infos_val.pkl
│   ├── semantickitti_infos_test.pkl

```

You can generate *.pkl by excuting

```
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
```

## Training and testing

```bash
# train
sh dist_train.sh $CONFIG $GPUS

# val
sh dist_test.sh $CONFIG $CHECKPOINT $GPUS

# test
sh dist_test.sh $CONFIG $CHECKPOINT $GPUS

```


## Acknowledgements

This work is based on the official codebase of [P3Former](https://github.com/OpenRobotLab/P3Former), we thank the contributors for their great work.
