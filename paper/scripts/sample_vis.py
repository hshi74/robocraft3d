import open3d as o3d

import copy
import glob
import numpy as np
import os
import rosbag
import ros_numpy
import sys
import trimesh

from config.config import gen_args
from datetime import datetime
from perception.pcd_utils import *
from perception.sample import *
from pysdf import SDF
from timeit import default_timer as timer
from transforms3d.quaternions import *
from tqdm import tqdm
from utils.data_utils import *
from utils.visualize import *


cd = os.path.dirname(os.path.realpath(sys.argv[0]))
figure_root = os.path.join(cd, '..', 'images')

# @profile
def sample(args, pcd, tool_list, use_vg_filter=False, visualize='o3d'):
    cube, rest = preprocess_raw_pcd(args, pcd, visualize=visualize=='o3d', rm_stats_outliers=2)

    if visualize == 'plt':
        render_o3d([pcd], point_size_list=[40], path=os.path.join(figure_root, f'plt_pcd'))
        render_o3d([cube], point_size_list=[40], path=os.path.join(figure_root, f'plt_cube'))

    cube_colors = np.asarray(cube.colors)
    color_avg = list(np.mean(cube_colors, axis=0))

    ##### 1. random sample 100x points in the bounding box #####
    lower = cube.get_min_bound()
    upper = cube.get_max_bound()
    sample_size = 10 * args.n_particles
    sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

    # TODO: Verify if this is a good choice
    # z_extent = (cube.get_max_bound() - cube.get_min_bound())[2]
    # if z_extent < 0.02:
    # selected_mesh = alpha_shape_mesh_reconstruct(cube, alpha=0.2, mesh_fix=False, visualize=visualize=='o3d')
   
    # f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    # else:
    selected_mesh = poisson_mesh_reconstruct(cube, depth=6, mesh_fix=False, visualize=False)
    
    # selected_mesh = o3d.geometry.TriangleMesh()
    # selected_mesh.vertices = o3d.utility.Vector3dVector(selected_mesh_old.points)
    # selected_mesh.triangles = o3d.utility.Vector3iVector(selected_mesh_old.faces.reshape(selected_mesh_old.n_faces, -1)[:, 1:])
    f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    # f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])

    if visualize == 'plt':
        render_o3d([selected_mesh], path=os.path.join(figure_root, f'plt_alpha_mesh'))

    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    if use_vg_filter:
        vg_mask = vg_filter(pcd, sampled_points, visualize=False)

        outlier_pcd = o3d.geometry.PointCloud()
        outliers = sampled_points[~vg_mask]
        outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
        outlier_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        if visualize == 'o3d':
            visualize_o3d([sampled_pcd, outlier_pcd], title='vg_filter_points')
        else:
            render_o3d([sampled_pcd, outlier_pcd], label_list=['dough', 'outliers'], point_size_list=[160, 160], path=os.path.join(figure_root, f'plt_vg_filter'))

        sampled_points = sampled_points[vg_mask]

        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    ##### 6. filter out the noise #####
    # sampled_pcd = o3d.geometry.PointCloud()
    # sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    # sampled_pcd = sampled_pcd.voxel_down_sample(voxel_size=0.002)

    cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    sampled_pcd_stat = sampled_pcd.select_by_index(inlier_ind_stat)
    outliers_stat = sampled_pcd.select_by_index(inlier_ind_stat, invert=True)

    sampled_pcd = sampled_pcd_stat

    cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    sampled_pcd_stat = sampled_pcd.select_by_index(inlier_ind_stat)
    outliers_stat = sampled_pcd.select_by_index(inlier_ind_stat, invert=True)
    
    sampled_pcd = sampled_pcd_stat
    
    if visualize == 'o3d':
        visualize_o3d([sampled_pcd], title='sampled_points')
    else:
        render_o3d([sampled_pcd], label_list=['dough'], point_size_list=[160], path=os.path.join(figure_root, f'plt_sample_points'))

    ##### 3. use SDF to filter out points INSIDE the tool mesh #####
    out_tool_points, in_tool_points = inside_tool_filter(sampled_points, tool_list, in_d=-0.001, visualize=visualize=='o3d')
    
    out_tool_pcd = o3d.geometry.PointCloud()
    out_tool_pcd.points = o3d.utility.Vector3dVector(out_tool_points)
    out_tool_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    in_tool_pcd = o3d.geometry.PointCloud()
    in_tool_pcd.points = o3d.utility.Vector3dVector(in_tool_points)
    in_tool_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    if visualize == 'o3d':
        visualize_o3d([out_tool_pcd, in_tool_pcd, *list(zip(*tool_list))[1]], title='tool_filter')
    else:
        render_o3d([out_tool_pcd, in_tool_pcd, *list(zip(*tool_list))[1]], label_list=['dough', 'outliers', 'tool', 'tool'], 
            point_size_list=[160, 160, 160, 160], axis_off=False, path=os.path.join(figure_root, f'plt_tool_filter'))

        # render_o3d([*list(zip(*tool_list))[1]], label_list=['tool'], 
        #     point_size_list=[160], axis_off=True, path=os.path.join(figure_root, f'plt_tool_filter'))

    sampled_pcd = out_tool_pcd
    # print(f'points touching: {n_points_touching}')
    # print(f'is_moving_back: {is_moving_back}')



    # if visualize:
    #     sampled_pcd.paint_uniform_color([0.0, 0.8, 0.0])
    #     outliers.paint_uniform_color([0.8, 0.0, 0.0])
    #     visualize_o3d([cube, sampled_pcd, outliers], title='cleaned_point_cloud', pcd_color=color_avg)

    ##### (optional) 8. surface sampling #####
    if args.surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(sampled_pcd, alpha=0.005, visualize=visualize=='o3d')
        if visualize == 'plt':
            render_o3d([selected_mesh], path=os.path.join(figure_root, f'plt_surf_mesh'))
        
        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, args.n_particles)
        surface_points = np.asarray(selected_surface.points)
        
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        if visualize == 'o3d':
            visualize_o3d([surface_pcd], title='surface_point_cloud', pcd_color=color_avg)
        else:
            surface_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            render_o3d([surface_pcd, *list(zip(*tool_list))[1]], point_size_list=[160, 160, 160], label_list=['dough', 'tool', 'tool'], 
                axis_off=False, path=os.path.join(figure_root, f'plt_surf_pcd'))

        selected_pcd = surface_pcd
    else:
        raise NotImplementedError

    return sampled_pcd, selected_pcd


# @profile
def ros_bag_to_pcd(args, bag_path, tool_list, use_vg_filter=False, visualize=False):
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

    fingertip_mat = quat2mat(ee_quat) @ args.ee_fingertip_T_mat[:3, :3]
    fingermid_pos = (quat2mat(ee_quat) @ args.ee_fingertip_T_mat[:3, 3].T).T + ee_pos

    tool_name_list = args.tool_geom_mapping[args.env]
    tool_list_T = []
    fingertip_T_list = []
    for k in range(len(tool_name_list)):
        if 'gripper' in args.env:
            fingertip_pos = (fingertip_mat @ np.array([0, (2 * k - 1) * (gripper_width) / 2, 0]).T).T + fingermid_pos
        else:
            fingertip_pos = fingermid_pos
        fingertip_T = np.concatenate((np.concatenate((fingertip_mat, np.array([fingertip_pos]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
        fingertip_T_list.append(fingertip_T)

        tool_mesh_T = copy.deepcopy(tool_list[k][0]).transform(fingertip_T)
        tool_surface_T = copy.deepcopy(tool_list[k][1]).transform(fingertip_T)
        tool_list_T.append((tool_mesh_T, tool_surface_T))

    pcd = merge_point_cloud(args, pcd_msgs, visualize=visualize=='o3d')

    # render_o3d([pcd], point_size_list=[40], axis_off=True, path=os.path.join(figure_root, os.path.basename(bag_path).replace('.bag', '')))

    sample(args, pcd, tool_list_T, use_vg_filter=use_vg_filter, visualize=visualize)

    # state_cur = np.asarray(pcd_sparse.points)
    # tool_repr = get_tool_repr(args, fingertip_T_list)
    # state_cur = np.concatenate([np.asarray(pcd_sparse.points), args.floor_state, tool_repr])

    # return pcd_dense, pcd_sparse, state_cur


def main():
    args = gen_args()
    args.env = 'gripper_sym_rod'
    tool_name_list = args.tool_geom_mapping[args.env]
    tool_list = []
    for i in range(len(tool_name_list)):
        tool_mesh = o3d.io.read_triangle_mesh(os.path.join(args.tool_geom_path, f'{tool_name_list[i]}.stl'))
        # tool_surface_dense = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 100000)
        # if 'press_circle' in tool_name_list[i]:
        #     voxel_size = 0.0057
        # else:
        #     voxel_size = 0.006
        # tool_surface = tool_surface_dense.voxel_down_sample(voxel_size=voxel_size)
        tool_surface_points = args.tool_full_repr_dict[args.env][i]
        tool_surface = o3d.geometry.PointCloud()
        tool_surface.points = o3d.utility.Vector3dVector(tool_surface_points)
        tool_surface.paint_uniform_color([1.0, 0.0, 0.0])
        tool_list.append((tool_mesh, tool_surface))

    write_frames = False

    ros_bag_path = 'paper/data/scene.bag'

    # ros_bag_path = "/media/hshi74/Game Drive PS4/robocook/raw_data/roller_large_robot_v4/ep_067/seq_000/9.071.bag"
    # ros_bag_path = "/media/hshi74/Game Drive PS4/robocook/raw_data/roller_large_robot_v4/ep_067/seq_000/15.119.bag"
    # rollout_path = os.path.dirname(ros_bag_path)
    # ros_bag_path_list = sorted(glob.glob("/media/hshi74/Game Drive PS4/robocook/raw_data/roller_large_robot_v4/ep_067/seq_000/*.bag"))
    # for ros_bag_path in ros_bag_path_list:
    ros_bag_to_pcd(args, ros_bag_path, tool_list, use_vg_filter=False, visualize='plt')

    # if write_frames:
    #     render_frames(args, ['Perception'], [np.array([state_cur])], views=[(90, 0)], path=os.path.join(rollout_path))


if __name__ == '__main__':
    main()
