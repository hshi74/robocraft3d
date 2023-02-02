import cv2 as cv
import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import open3d as o3d
import os
import pandas as pd
import pickle
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)
import sys
import torch
import torchvision.transforms as transforms

from datetime import datetime
from sklearn import metrics
from trimesh import PointCloud

matplotlib.rcParams["legend.loc"] = 'lower right'
color_list = ['royalblue', 'red', 'green', 'cyan', 'orange', 'pink', 'tomato', 'violet']


def plot_ablation(data, x, path=''):
    plt.figure(figsize=(8, 6))
    plt.gca().set_prop_cycle(None)
    for name in ['CD', 'EMD']:
        ax = sns.lineplot(data=data, x=x, y=f'{name}_avg', linewidth=5)

        ax.set(xlabel=None)
        ax.set(ylabel=None)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        loss_min_bound = data[f'{name}_avg'] - data[f'{name}_std']
        loss_max_bound = data[f'{name}_avg'] + data[f'{name}_std']
        plt.fill_between(data[x], loss_max_bound, loss_min_bound, alpha=0.3)

    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(f'{path}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
    else:
        plt.show()

    plt.close()


def main():
    for x in ['sequence_length', 'neighbor_radius', 'tool_neighbor_radius', 'weight']:
        data = pd.read_csv(f'paper/data/metrics - {x}.csv')
        plot_ablation(data, x, path=f'paper/images/{x}')


if __name__ == "__main__":
    main()