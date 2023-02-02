import glob
import numpy as np
import open3d as o3d
import os
import sys

from perception.pcd_utils import *
from perception.sample import *
from utils.config import gen_args
from utils.data_utils import *
from utils.visualize import *

def process_pcd(args, bag_path, n_particles=4096, visualize=False):
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

    if 'hook' in bag_path or 'spatula_large' in bag_path or ('spatula_small' in bag_path and 'out.bag' in bag_path):
        pcd = merge_point_cloud(args, pcd_msgs, crop_range=[-0.085, -0.26, 0.0, 0.065, -0.11, 0.07], visualize=visualize)
    else:
        pcd = merge_point_cloud(args, pcd_msgs, crop_range=[-0.075, -0.075, 0.0, 0.075, 0.075, 0.07], visualize=visualize)

    # cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize, rm_stats_outliers=2)
    dough = pcd

    # dough_points = np.asarray(cube.points)
    # dough_colors = np.asarray(cube.colors)

    # if len(dough_points) > 0:
    #     farthest_pts = np.zeros((n_particles, 3))
    #     farthest_colors = np.zeros((n_particles, 3))
    #     first_idx = np.random.randint(len(dough_points))
    #     farthest_pts[0] = dough_points[first_idx]
    #     farthest_colors[0] = dough_colors[first_idx]
    #     distances = calc_distances(farthest_pts[0], dough_points)
    #     for i in range(1, n_particles):
    #         next_idx = np.argmax(distances)
    #         farthest_pts[i] = dough_points[next_idx]
    #         farthest_colors[i] = dough_colors[next_idx]
    #         distances = np.minimum(distances, calc_distances(farthest_pts[i], dough_points))

    #     dough_points = farthest_pts
    #     dough_colors = farthest_colors

    # dough = o3d.geometry.PointCloud()
    # dough.points = o3d.utility.Vector3dVector(dough_points)
    # dough.colors = o3d.utility.Vector3dVector(dough_colors)

    return dough


def main():
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    args = gen_args()
    visualize = False

    state_root = "/media/hshi74/Game Drive PS4/robocook/raw_data/classifier_08-24/ep_709/seq_012_hook/"
    state_in = process_pcd(args, state_root + 'in.bag', visualize=visualize)
    state_out = process_pcd(args, state_root + 'out.bag', visualize=visualize)
    state_list = []
    for state_pcd in [state_in, state_out]:
        state = np.concatenate((np.asarray(state_pcd.points), np.asarray(state_pcd.colors)), axis=1)
        state_list.append(state)
    visualize_pcd_pred(['IN', 'OUT'], state_list, res='high', path=os.path.join(cd, '..', 'images', 'cls_2.png'))


if __name__ == "__main__":
    main()