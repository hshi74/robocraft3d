import glob
import numpy as np
import open3d as o3d
import os

from planning.pcd_tool_classifier.build_dataset import process_pcd
from planning.image_tool_classifier.build_dataset import crop_image
from PIL import Image
from perception.pcd_utils import *
from perception.get_visual_feedback import ros_bag_to_pcd
from config.config import gen_args
from utils.data_utils import *
from utils.visualize import *


def crop(img_dir, dim=200):
    im = Image.open(img_dir)
    width, height = im.size
    if width > dim or height > dim:
        cropped_img = crop_image(img_dir, dim=dim, visualize=False)
        cropped_img.save(img_dir)


def main():
    args = gen_args()
    debug = False
    visualize = False
    target_type = 'alphabet_real'

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    target_dir = os.path.join(cd, '..', '..', 'target_shapes', target_type)
    test_dir_list = sorted(glob.glob(os.path.join(target_dir, '*')))
    target_set = ['R', 'O', 'B', 'C', 'K'] 

    for i in range(len(test_dir_list)): 
        if debug and i > 0: break
        test_dir = test_dir_list[i]
        if os.path.basename(test_dir) in target_set:
            print(test_dir)
            subtarget_list = [d.name for d in os.scandir(test_dir) if d.is_dir()]
            for subtarget in subtarget_list:
                step = subtarget.split('_')[0]
                tool_name = '_'.join(subtarget.split('_')[1:])
                print(tool_name)
                # if os.path.exists(os.path.join(test_dir, subtarget, f'{step}.h5')): continue

                img_paths = sorted(glob.glob(os.path.join(test_dir, subtarget, '*cam*.png')))
                for img_dir in img_paths:
                    # print(img_dir)
                    crop(img_dir)

                ros_bag_path = os.path.join(test_dir, subtarget, f'{step}.bag')

                cube = process_pcd(args, ros_bag_path, visualize=visualize)
                o3d.io.write_point_cloud(os.path.join(test_dir, subtarget, f'{step}_raw.ply'), cube)

                args.env = tool_name
                pcd_dense, pcd_sparse, _, = ros_bag_to_pcd(args, ros_bag_path, use_vg_filter=True, visualize=visualize)
                
                h5_path = os.path.join(test_dir, subtarget, f'{step}.h5')
                h5_dense_path = os.path.join(test_dir, subtarget, f'{step}_dense.h5')
                h5_surf_path = os.path.join(test_dir, subtarget, f'{step}_surf.h5')
                
                # state = np.concatenate((np.asarray(pcd_sparse.points), args.floor_state))
                # state_dense = np.concatenate((np.asarray(pcd_dense.points), args.floor_state))
                state = np.asarray(pcd_sparse.points)
                state_dense = np.asarray(pcd_dense.points)

                surf_mesh = alpha_shape_mesh_reconstruct(pcd_dense, alpha=0.005, visualize=visualize)
                state_surf_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                    surf_mesh, args.n_particles)
                # state_surf = np.concatenate((np.asarray(state_surf_pcd.points), args.floor_state))
                state_surf = np.asarray(state_surf_pcd.points)

                args.env = 'gripper_sym_rod'
                shape_quats = np.zeros((args.floor_dim + sum(args.tool_dim[args.env]), 4), dtype=np.float32)
                
                h5_data = [state, shape_quats, args.scene_params]
                h5_data_dense = [state_dense, shape_quats, args.scene_params]
                h5_data_surf = [state_surf, shape_quats, args.scene_params]

                store_data(args.data_names, h5_data, h5_path)
                store_data(args.data_names, h5_data_dense, h5_dense_path)
                store_data(args.data_names, h5_data_surf, h5_surf_path)

                render_frames(args, [f'{step}', f'{step}_dense', f'{step}_surf'], [np.array([h5_data[0]]), np.array([h5_data_dense[0]]), 
                    np.array([h5_data_surf[0]])], axis_off=False, path=os.path.join(test_dir, subtarget), name=f'{step}_sampled.png')


if __name__ == "__main__":
    main()
