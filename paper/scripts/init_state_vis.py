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


# @profile
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

    cube, rest = preprocess_raw_pcd(args, pcd, visualize=False, rm_stats_outliers=2)

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
    selected_mesh = alpha_shape_mesh_reconstruct(cube, alpha=0.2, mesh_fix=False, visualize=False)

    f = SDF(selected_mesh.vertices, selected_mesh.triangles)
    # else:
    #     selected_mesh = poisson_mesh_reconstruct(cube, depth=6, mesh_fix=True, visualize=visualize)
    #     f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])

    sdf = f(sampled_points)
    sampled_points = sampled_points[sdf > 0]
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    # if use_vg_filter:
    #     vg_mask = vg_filter(pcd, sampled_points, visualize=False)

    #     outlier_pcd = o3d.geometry.PointCloud()
    #     outliers = sampled_points[~vg_mask]
    #     outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
    #     outlier_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    #     if visualize == 'o3d':
    #         visualize_o3d([sampled_pcd, outlier_pcd], title='vg_filter_points')
    #     else:
    #         render_o3d([sampled_pcd, outlier_pcd], label_list=['dough', 'outliers'], point_size_list=[160, 160], path=os.path.join(figure_root, f'plt_vg_filter'))

    #     sampled_points = sampled_points[vg_mask]

    #     sampled_pcd = o3d.geometry.PointCloud()
    #     sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    #     sampled_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    ##### 3. use SDF to filter out points INSIDE the tool mesh #####
    # out_tool_points, in_tool_points = inside_tool_filter(sampled_points, tool_list, in_d=-0.001, visualize=False)
    
    # out_tool_pcd = o3d.geometry.PointCloud()
    # out_tool_pcd.points = o3d.utility.Vector3dVector(out_tool_points)
    # out_tool_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    # in_tool_pcd = o3d.geometry.PointCloud()
    # in_tool_pcd.points = o3d.utility.Vector3dVector(in_tool_points)
    # in_tool_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    # sampled_pcd = out_tool_pcd

    ##### (optional) 8. surface sampling #####
    if args.surface_sample:
        ##### 7. farthest point sampling to downsample the pcd to 300 points #####
        # fps_points = fps(np.asarray(sampled_pcd.points), args.n_particles * 10)
        # fps_pcd = o3d.geometry.PointCloud()
        # fps_pcd.points = o3d.utility.Vector3dVector(fps_points)

        selected_mesh = alpha_shape_mesh_reconstruct(sampled_pcd, alpha=0.005, visualize=False)
        # if visualize == 'plt':
        #     render_o3d([selected_mesh], path=os.path.join(figure_root, f'plt_surf_mesh'))
        
        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, args.n_particles)
        surface_points = np.asarray(selected_surface.points)
        
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        selected_pcd = surface_pcd
    else:
        raise NotImplementedError

    return sampled_pcd, selected_pcd


def main():
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    args = gen_args()
    visualize = False

    root = "/media/hshi74/Game Drive PS4/robocook/raw_data/gripper_sym_rod_robot_v4"

    for idx in range(10):
        state_root = root + f"/ep_{str(idx).zfill(3)}/seq_000/"
        _, state_pcd = process_pcd(args, state_root + '0.000.bag', visualize=visualize)
        state = np.asarray(state_pcd.points)
        visualize_pcd_pred(['IN'], [state], res='high', axis_off=True, path=os.path.join(cd, '..', 'images', f'init_state_{idx}.png'))


if __name__ == "__main__":
    main()