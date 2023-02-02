import open3d as o3d

import copy
import csv
import glob
import numpy as np
import os
import rosbag
import ros_numpy
import sys
import trimesh

from datetime import datetime
from perception.pcd_utils import *
from perception.sample import *
from pysdf import SDF
from timeit import default_timer as timer
from transforms3d.quaternions import *
from tqdm import tqdm
from utils.config import gen_args
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


cd = os.path.dirname(os.path.realpath(sys.argv[0]))
figure_root = os.path.join(cd, '..', 'images')

# @profile
def sample(args, pcd, use_vg_filter=False, visualize='o3d'):
    cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize=='o3d', rm_stats_outliers=2)

    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    ##### 1. random sample 100x points in the bounding box #####
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    sample_size = 100 * args.n_particles
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    # TODO: Verify if this is a good choice
    # z_extent = (cube.get_max_bound() - cube.get_min_bound())[2]
    # if z_extent < 0.02:
    selected_mesh = alpha_shape_mesh_reconstruct(cube, alpha=0.2, mesh_fix=False, visualize=visualize=='o3d')

    f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    # else:
    #     selected_mesh = poisson_mesh_reconstruct(cube, depth=6, mesh_fix=True, visualize=visualize)
    #     f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])

    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    if visualize == 'o3d':
        visualize_o3d([sampled_pcd], title='sampled_points')

    if use_vg_filter:
        vg_mask = vg_filter(pcd, sampled_points, visualize=False)

        outlier_pcd = o3d.geometry.PointCloud()
        outliers = sampled_points[~vg_mask]
        outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
        outlier_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        if visualize == 'o3d':
            visualize_o3d([sampled_pcd, outlier_pcd], title='vg_filter_points')

        sampled_points = sampled_points[vg_mask]

        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    ##### (optional) 8. surface sampling #####
    if args.surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(sampled_pcd, alpha=0.005, visualize=visualize=='o3d')
        
        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, args.n_particles)
        surface_points = np.asarray(selected_surface.points)
        
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        if visualize == 'o3d':
            visualize_o3d([surface_pcd], title='surface_point_cloud', pcd_color=color_avg)

        selected_pcd = surface_pcd
    else:
        raise NotImplementedError

    return sampled_pcd, selected_pcd


# @profile
def ros_bag_to_pcd(args, bag_path, use_vg_filter=False, visualize=False):
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

    ee_pos, ee_quat, gripper_width = None, None, None
    for topic, msg, t in bag.read_messages(
        topics=['/cam1/depth/color/points', '/cam2/depth/color/points', '/cam3/depth/color/points', '/cam4/depth/color/points', 
        '/ee_pose', '/gripper_width']
    ):
        if 'cam' in topic:
            pcd_msgs.append(msg)

        if topic == '/ee_pose':
            ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
            ee_quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        
        if topic == '/gripper_width':
            gripper_width = msg.data

    bag.close()

    if 'hook' in args.env or 'spatula_large' in args.env or 'spatula_small' in args.env:
        pcd = merge_point_cloud(args, pcd_msgs, crop_range=[-0.1, -0.25, 0.02, 0.1, -0.05, 0.08], visualize=visualize=='o3d')
    else:
        pcd = merge_point_cloud(args, pcd_msgs, crop_range=[-0.1, -0.1, 0.005, 0.1, 0.1, 0.15], visualize=visualize=='o3d')

    # render_o3d([pcd], point_size_list=[40], axis_off=True, path=os.path.join(figure_root, f'pcd', os.path.basename(bag_path).replace('.bag', '')))

    sampled_pcd, selected_pcd = sample(args, pcd, use_vg_filter=use_vg_filter, visualize=visualize)

    pcd_path = bag_path.replace('.bag', '.ply')
    o3d.io.write_point_cloud(pcd_path, selected_pcd)

    # state_cur = np.asarray(pcd_sparse.points)
    # tool_repr = get_tool_repr(args, fingertip_T_list)
    # state_cur = np.concatenate([np.asarray(pcd_sparse.points), args.floor_state, tool_repr])

    # return pcd_dense, pcd_sparse, state_cur


def main():
    args = gen_args()

    # bag_file_path_list = sorted(glob.glob('paper/data/*.bag'))
    # for ros_bag_path in bag_file_path_list:
    #     if 'B' in ros_bag_path: continue
    #     ros_bag_to_pcd(args, ros_bag_path, use_vg_filter=True, visualize='ff')

    # pkl_file_path_list = sorted(glob.glob('paper/data/*.pkl'))
    # for pickle_path in pkl_file_path_list:
    #     with open(pickle_path, 'rb') as f:
    #         args_dict = pickle.load(f)
    #         state_final = args_dict['state_seqs'][0][-1, :args.n_particles]

    #     pcd_path = pickle_path.replace('.pkl', '.ply')
    #     selected_pcd = o3d.geometry.PointCloud()
    #     selected_pcd.points = o3d.utility.Vector3dVector(state_final)
    #     o3d.io.write_point_cloud(pcd_path, selected_pcd)

    # ply_file_list = sorted(glob.glob('paper/data/*.ply'))
    # bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, 0.015], [1, 1, 1])
    # for ply_file_path in ply_file_list:
    #     if 'crop' in ply_file_path: continue
    #     pcd = o3d.io.read_point_cloud(ply_file_path)
    #     # points = np.asarray(pcd.points)
    #     # normals = get_normals(points[None])[0]
    #     # pcd.normals = o3d.utility.Vector3dVector(normals)

    #     # o3d.io.write_point_cloud(ply_file_path, pcd)
    #     # hist, bin_edges = np.histogram(points[:, 2], bins=3)
    #     # print(hist, bin_edges)
    #     pcd_crop = pcd.crop(bbox)

    #     # visualize_o3d([pcd], title=os.path.basename(ply_file_path), show_normal=True)
    #     o3d.io.write_point_cloud(ply_file_path.split('.ply')[0] + '_crop.ply', pcd_crop)


    header = ['letter', 'method', 'chamfer', 'emd', 'normal']

    with open('paper/data/loss.csv', 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

        ply_file_list = sorted(glob.glob('paper/data/*.ply'))
        for ply_file_path in ply_file_list:
            if 'crop' in ply_file_path: continue
            # if not 'O_L' in ply_file_path: continue

            name_list = os.path.basename(ply_file_path).split('_')
            target_letter = name_list[0]
            method_name = name_list[1].split('.')[0]
            data = [target_letter, method_name]

            pcd = o3d.io.read_point_cloud(ply_file_path)
            pcd.paint_uniform_color([0, 0, 1])
            full_points = np.asarray(pcd.points)
            full_normals = get_normals(full_points[None])[0]
            pcd.normals = o3d.utility.Vector3dVector(full_normals)

            target_points = load_data(args.data_names, f'paper/data/{target_letter}_target.h5')[0]
            target_normals = get_normals(target_points[None])[0]
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            target_pcd.normals = o3d.utility.Vector3dVector(target_normals)
            target_pcd.paint_uniform_color([0, 1, 0])


            full_points_norm = full_points - np.mean(full_points, axis=0)
            target_points_norm = target_points - np.mean(target_points, axis=0)

            # pcd_norm = o3d.geometry.PointCloud()
            # pcd_norm.points = o3d.utility.Vector3dVector(full_points_norm)
            # pcd_norm.paint_uniform_color([0, 0, 1])

            # target_pcd_norm = o3d.geometry.PointCloud()
            # target_pcd_norm.points = o3d.utility.Vector3dVector(target_points_norm)
            # target_pcd_norm.paint_uniform_color([0, 1, 0])

            # visualize_o3d([pcd_norm, target_pcd_norm])

            cd = chamfer(full_points_norm, target_points_norm)
            data.append(cd)

            e = emd(full_points_norm, target_points_norm)
            data.append(e)

            full_hist, full_bin_edges = np.histogram(full_points[:, 2], bins=4)
            full_bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, full_bin_edges[-3]], [1, 1, 1])

            target_hist, target_bin_edges = np.histogram(target_points[:, 2], bins=4)
            target_bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, target_bin_edges[-3]], [1, 1, 1])

            pcd_crop = pcd.crop(full_bbox)
            full_center = pcd_crop.get_center()
            full_center[2] = 1
            pcd_crop.orient_normals_towards_camera_location(full_center)
            target_pcd_crop = target_pcd.crop(target_bbox)
            target_center = target_pcd_crop.get_center()
            target_center[2] = 1
            target_pcd_crop.orient_normals_towards_camera_location(target_center)

            # visualize_o3d([pcd], show_normal=True)

            normal_cd = chamfer(np.asarray(pcd_crop.normals), np.asarray(target_pcd_crop.normals))
            data.append(normal_cd)

            writer.writerow(data)


if __name__ == '__main__':
    main()
