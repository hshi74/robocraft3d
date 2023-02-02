import glob
import numpy as np
import open3d as o3d
import os
import subprocess
import sys

from config.config import gen_args
from perception.pcd_utils import *
from perception.get_visual_feedback import *
from utils.data_utils import *
from utils.visualize import *

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

##### TO CREATE A SERIES OF PICTURES

def make_views(ax,angles,elevation=30):
    for i,angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = f'paper/images/tmprot_{str(i).zfill(3)}.png'
        # plt.show()
        ax.figure.savefig(fname)


##### MAIN FUNCTION

def rotanimate(ax, angles, output, **kwargs):
    make_views(ax,angles, **kwargs)

    subprocess.run(['ffmpeg', '-y', '-i', f'paper/images/tmprot_%03d.png', '-c:v', 'libx264', 
        '-vf', 'fps=30', '-pix_fmt', 'yuv420p', output], 
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    subprocess.run(f'rm paper/images/tmprot_*.png', shell=True)


def process_pcd(args, bag_path, visualize=False):
    pcd_msgs = []
    while True:
        try:
            bag = rosbag.Bag(bag_path) # allow_unindexed=True
            break
        except rosbag.bag.ROSBagUnindexedException:
            print('Reindex the rosbag file:')
            os.system(f"rosbag reindex {bag_path}")
            bag_orig_path = os.path.join(os.path.dirname(bag_path), 'pcd.orig.bag') 
            os.system(f"rm {bag_orig_path}")
        except rosbag.bag.ROSBagException:
            continue

    for topic, msg, t in bag.read_messages(
        topics=['/cam1/depth/color/points', '/cam2/depth/color/points', '/cam3/depth/color/points', '/cam4/depth/color/points', 
        '/ee_pose', '/gripper_width']
    ):
        if 'cam' in topic:
            pcd_msgs.append(msg)

    bag.close()

    pcd = merge_point_cloud(args, pcd_msgs, visualize=visualize)

    cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize)

    # pcd_colors = np.asarray(pcd.colors, dtype=np.float32)
    # # bgr
    # pcd_rgb = pcd_colors[None, :, :]

    # pcd_hsv = cv.cvtColor(pcd_rgb, cv.COLOR_RGB2HSV)
    # hsv_lower = np.array([0, 0.5, 0], dtype=np.float32)
    # hsv_upper = np.array([360, 1, 1], dtype=np.float32)
    # mask = cv.inRange(pcd_hsv, hsv_lower, hsv_upper)
    # cube_label = np.where(mask[0] == 255)

    # cube = pcd.select_by_index(cube_label[0])

    # cl, inlier_ind_cube_stat = cube.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    # cube_stat = cube.select_by_index(inlier_ind_cube_stat)
    # cube = cube_stat

    if visualize:
        visualize_o3d([cube], title='selected_dough')

    return cube


def main():
    target_root = "target_shapes/3d_real"
    args = gen_args()
    visualize = False

    target_list = ['pagoda']

    for target_name in target_list:
        # pcd_dense, pcd_sparse, state_cur = ros_bag_to_pcd(args, os.path.join(target_root, target_name, f'001.bag'), visualize=visualize)

        # cube = pcd_sparse
        
        cube = process_pcd(args, os.path.join(target_root, target_name, f'001.bag'), visualize=visualize)

        fig = plt.figure(figsize=(12, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_facecolor("black")
        cube_points = np.asarray(cube.points)
        cube_colors = np.asarray(cube.colors)
        X, Y, Z = cube_points[:, 0], cube_points[:, 1], cube_points[:, 2]

        # ax.plot_trisurf(X, Y, Z, linewidth=0.5, antialiased=True)

        ax.scatter(X, Y, Z, c=cube_colors, s=30)
        # plt.axis('off') # remove axes for visual appeal
        # ax.set_xlabel('X', labelpad=20, fontsize=30)
        # ax.set_ylabel('Y', labelpad=20, fontsize=30)
        # ax.set_zlabel('Z', labelpad=20, fontsize=30)

        # ax.tick_params(axis='x', labelsize='large', pad=10)
        # ax.tick_params(axis='y', labelsize='large', pad=10)
        # ax.tick_params(axis='z', labelsize='large', pad=10)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.tight_layout()

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect((max_range * 2, max_range * 2, max_range * 2))
        
        angles = np.linspace(0,360,361)[:-1] # Take 20 angles between 0 and 360
    
        rotanimate(ax, angles,f'paper/images/{target_name}_movie.mp4')

        plt.close()


if __name__ == "__main__":
    main()