
<img src='imgs/robocraft.gif' align="right" width=384>

<br><br><br>

# RoboCraft: Learning to See, Simulate, Shape Elasto-Plastic Object with Graph Networks

**RoboCraft: [Project](https://sites.google.com/view/robocraftplasticine/home) |  [Paper]()**

<img src="xxx.jpg" width="800"/>


If you use this code for your research, please cite:

RoboCraft: Learning to See, Simulate, Shape Elasto-Plastic Object with Graph Networks<br>
[Haochen Shi]()\*,  [Huazhe Xu](https://hxu.rocks)\*, [Yunzhu Li](https://people.csail.mit.edu/liyunzhu/), [Zhiao Huang](https://sites.google.com/view/zhiao-huang), [Jiajun Wu](https://jiajunwu.com/). In xxxx 2022. (* equal contributions) [[Bibtex]](https://hxu.rocks/robocraft/robocraft.txt)


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/hshi74/robocook.git
cd RoboCraft
conda env create -f robocraft.yml
conda activate robocraft
```

- Install requirements for the simulator.
  - For pip users, please type the command `cd simulator` and `pip install -e .`.

- Go to the simulator dir
  - `cd ./plb/algorithms`

- Add this line to your ~/.bashrc
  - `export PYTHONPATH="${PYTHONPATH}:[path/to/robocook]"`

### Data Generation
- Run all the blocks in `test_tasks.ipynb`. We note that it is much easier to use ipython notebook when dealing with Taichi env for fast materialization.

### Particle Sampling
- run `python sample_data.py`

### Prepare for Dynamics Model
```bash
cd ../../../dynamics
bash scripts/utils/move_data.sh ngrip_fixed sample_ngrip_fixed_14-Feb-2022-21:24:27.516157
```

### Train Dynamics Model
Run `bash scripts/dynamics/train.sh`

### Planning with the Learned Model
Run `bash scripts/control/control.sh`

## Code structure
The simulator folder contains the simulation environment we used for data collection and particle sampling. 
The dynamics folder contains the code for learning the GNN and planning.
The models folder contains the code for all the 3D printed gadgets.
The franka_robot folder contains the code used for real world control.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{robocraft,
  title={Learning to See, Simulate, Shape Elasto-Plastic Objects with Graph Networks},
  author={xxxx},
  booktitle={xxxx},
  year={xxxx}
}
```
