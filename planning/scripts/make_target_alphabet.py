import cv2
import numpy as np
import open3d as o3d
import os

from perception.pcd_utils import *
from pysdf import SDF
from sdf import *
from string import ascii_uppercase
from config.config import gen_args
from utils.data_utils import *
from utils.visualize import *


def render_alphabet(args, state_seq, draw_set=['dough','floor'], view=(90, -90), path=''):
    n_rows = 4
    n_cols = 7

    fig, big_axes = plt.subplots(n_rows, 1, figsize=(3 * n_cols, 3 * n_rows))

    for i in range(n_rows):
        big_axes[i].axis('off')
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            ax.view_init(*view)
            try:
                visualize_points(ax, args, state_seq[idx], draw_set, None)
            except IndexError:
                break

    plt.tight_layout()
        
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def main():
    args = gen_args()
    update = True
    debug = True
    target_type = 'alphabet_sim'
    suffix = ''

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    target_dir = os.path.join(cd, '..', '..', 'target_shapes', target_type)
    if 'alphabet' in target_type:
        if debug:
            image_names = ['R']
        else:
            image_names = list(ascii_uppercase)
    else:
        raise NotImplementedError

    h = 0.022
    shape_size = (0.065**2, h) # m^3
    shape_pos = (0.4, -0.1, 0.005 + h / 2)
    state_seq = []
    size_ratio_list = []
    for name in image_names:
        target_path = os.path.join(target_dir, name)
        pcd_path = os.path.join(target_path, f'{name}{suffix}.ply')
        if not os.path.exists(pcd_path) or update:
            image_path = os.path.join(target_path, f'{name}{suffix}.png')
            scaled_image_path = os.path.join(target_path, f'{name}{suffix}_scaled.png')
            
            image_ori = cv2.bitwise_not(cv2.imread(image_path))
            image_ori = cv2.copyMakeBorder(image_ori, 50, 50, 50, 50, cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwrite(scaled_image_path, image_ori)

            scaled_image = cv2.imread(scaled_image_path)
            gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
            pixels = cv2.countNonZero(thresh)
            image_area = scaled_image.shape[0] * scaled_image.shape[1]
            size_ratio = pixels / image_area
            size_ratio_list.append(size_ratio)
            print(f'size_ratio: {size_ratio}')

            scale_ratio = np.sqrt(shape_size[0] / 0.35)
            f = image(scaled_image_path).scale((scale_ratio, scale_ratio)).extrude(shape_size[1]).rotate(np.pi / 2).translate(shape_pos)
            f.save(pcd_path, step=0.001)

        h5_path = os.path.join(target_path, f'{name}{suffix}.h5')
        h5_dense_path = os.path.join(target_path, f'{name}{suffix}_dense.h5')
        h5_surf_path = os.path.join(target_path, f'{name}{suffix}_surf.h5')
        if not os.path.exists(h5_path) or update:
            tri_mesh = o3d.io.read_triangle_mesh(pcd_path)
            lower = tri_mesh.get_min_bound()
            upper = tri_mesh.get_max_bound()

            sample_size = 100 * args.n_particles
            sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower
        
            f = SDF(tri_mesh.vertices, tri_mesh.triangles)
            sdf = f(sampled_points)
            sampled_points = sampled_points[sdf > 0, :]
        
            sampled_pcd = o3d.geometry.PointCloud()
            sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        
            cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=150, std_ratio=1.5)
            sampled_pcd = sampled_pcd.select_by_index(inlier_ind)
            sampled_points_sparse = fps(np.asarray(sampled_pcd.points), args.n_particles)

            state = np.concatenate((sampled_points_sparse, args.floor_state))
            state_dense = np.concatenate((sampled_points, args.floor_state))

            surf_mesh = alpha_shape_mesh_reconstruct(sampled_pcd, alpha=0.01)
            state_surf_pcd = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                surf_mesh, args.n_particles)
            state_surf = np.concatenate((np.asarray(state_surf_pcd.points), args.floor_state))
            state_surf = np.asarray(state_surf_pcd.points)

            shape_quats = np.zeros((sum(args.tool_dim[args.env]) + args.floor_dim, 4), dtype=np.float32)
            h5_data = [state, shape_quats, args.scene_params]
            h5_data_dense = [state_dense, shape_quats, args.scene_params]
            h5_data_surf = [state_surf, shape_quats, args.scene_params]

            store_data(args.data_names, h5_data, h5_path)
            store_data(args.data_names, h5_data_dense, h5_dense_path)
            store_data(args.data_names, h5_data_surf, h5_surf_path)
        
        target_data = load_data(args.data_names, h5_path)
        # target_dense_data = load_data(args.data_names, h5_dense_path)
        # target_surf_data = load_data(args.data_names, h5_surf_path)
        state_seq.append(target_data[0])
        
        render_frames(args, ['Target'], [np.array([target_data[0]])], axis_off=False, 
            draw_set=['dough', 'floor'], path=target_path, name=f'{name}{suffix}_sampled.png')

        # render_frames(args, ['Target'], [np.array([target_dense_data[0]])], axis_off=False, 
        #     path=target_path, name=f'{name}{suffix}_dense_sampled.png')
        
        # render_frames(args, ['Target'], [np.array([target_surf_data[0]])], axis_off=False, 
        #     path=target_path, name=f'{name}{suffix}_surf_sampled.png')

    print(size_ratio_list)
    print(f"Average size ratio: {sum(size_ratio_list) / len(size_ratio_list)}")
    render_alphabet(args, state_seq, path=os.path.join(target_dir, 'alphabet_target.png'))


if __name__ == "__main__":
    main()
