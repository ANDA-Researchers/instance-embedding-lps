# ANDA-Vision Team

## Introduction

This is an official implementation of the paper "Instance Embedding as Queries for DETR-based LiDAR Panoptic Segmentation".


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
