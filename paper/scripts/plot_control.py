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


def main():
    path='paper/images/control'

    df = pd.DataFrame({
        'Methods': ['GD', 'CEM', 'RS', 'RL'],
        'CD': [0.0362, 0.0377, 0.0386, 0.0396],
        'EMD': [0.0285, 0.0301, 0.0312, 0.0318]
    })

    tidy = df.melt(id_vars='Methods').rename(columns=str.title)

    plt.figure(figsize=(8, 6))
    plt.gca().set_prop_cycle(None)

    ax = sns.barplot(data=tidy, x='Methods', y='Value', hue='Variable')

    ax.legend([],[], frameon=False)

    ax.set(xlabel=None)
    ax.set(ylabel=None)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(f'{path}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()