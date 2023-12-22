# RoboCraft: Learning to see, simulate, and shape elasto-plastic objects in 3D with graph networks

## Overview

**[Paper](https://doi.org/10.1177/02783649231219020)**

<img src="images/robocraft3d.gif" width="600">

## Prerequisites
- Linux or macOS (Tested on Ubuntu 20.04)
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Conda

## Getting Started

### Setup
```bash
# clone the repo
git clone https://github.com/hshi74/robocraft3d.git

# create the conda environment
conda env create -f robocraft.yml
conda activate robocraft

# install requirements for the simulator
pip install -e simulator

# install pytorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# add this line to your bashrc
export PYTHONPATH="${PYTHONPATH}:/path/to/robocraft3d"
```

### Generate Data
See [the controller codebase of RoboCraft](https://github.com/hshi74/deformable_ros)

### For all the following bash or python scripts, you will need to modify certain hyperparameters (like directories) before you run them.

### Sample particles and build the dataset for the GNN
1. `bash perception/scripts/run_sample.sh`.
1. (Optional) There could be a tiny portion of problematic datapoints. If you want to manually check the sampling results and remove the problematic ones, 
    1. Run `perception/scripts/inspect_perception.sh` to move all the visualizations into the same folder for your convenience
    1. Manually go through all the videos and type the indices of problematic videos into `dump/perception/inspect/inspect.txt`
    1. Run `perception/scripts/clean_perception.sh` to remove all the problemtaic ones.
1. Run `percetion/scripts/make_dataset.py` to build the dataset for the GNN

### Train Dynamics Model
`bash dynamics/scripts/run_train.sh`

### Planning with the Learned Model
`bash planning/scripts/run_control.sh`

## Code structure
- `config/`: config files for perception, dyanmics, planning, and simulation
- `dynamics/`: scripts to train and evaluate the GNN.
- `geometries/`: the STL files for tools and assets and surface point cloud representations for tools (in the `.npy` files)
- `models/`: a GNN checkpoint trained by us and its configurations (in the `.npy` file)
- `perception/`: the perception module of RoboCraft
- `planning/`: the planning module of RoboCraft
- `simulator/`: the simulation environment [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab)
- `target_shapes/`: point clouds of some target shapes
- `utils/`: utility and visualization functions

## Citation
If you use the codebase in your research, please cite:
```
@article{doi:10.1177/02783649231219020,
  author={Haochen Shi and Huazhe Xu and Zhiao Huang and Yunzhu Li and Jiajun Wu},
  title={RoboCraft: Learning to see, simulate, and shape elasto-plastic objects in 3D with graph networks},
  journal={The International Journal of Robotics Research},
  volume={0},
  number={0},
  pages={02783649231219020},
  year={2023},
  doi={10.1177/02783649231219020},
  URL={https://doi.org/10.1177/02783649231219020}
}
```
