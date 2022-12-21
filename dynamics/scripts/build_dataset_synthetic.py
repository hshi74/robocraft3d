import glob
import numpy as np
import os
import torch
import yaml

from dynamics.gnn import GNN
from perception.pcd_utils import alpha_shape_mesh_reconstruct
from tqdm import tqdm
from config.config import *
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


cd = os.path.dirname(os.path.realpath(sys.argv[0]))

def build_synthetic_dataset(args, tool_name, tool_model_path):
    print(tool_name, tool_model_path)
    tool_type = tool_model_path.split('/')[-2].replace('dump_', '')
    dy_dataset_root = os.path.join(cd, '..', '..', 'data', 'gt', f'data_{tool_type}')

    # tee = Tee(os.path.join(synthetic_dataset_root, 'dataset.txt'), 'w')

    tool_args = copy.deepcopy(args)
    tool_args.env = tool_name
    # dy_args_dict = np.load(f'{tool_model_path}_args.npy', allow_pickle=True).item()
    dy_args_dict = np.load(f'{tool_model_path}/args.npy', allow_pickle=True).item()
    tool_args = update_dy_args(tool_args, tool_name, dy_args_dict)

    synthetic_dataset_root = os.path.join(cd, '..', '..', 'data', 'synthetic', f'data_{tool_type}_time_step={tool_args.time_step}')
    os.system('mkdir -p ' + synthetic_dataset_root)

    # print_args(tool_args)
    # gnn = GNN(tool_args, f'{tool_model_path}.pth')
    gnn = GNN(tool_args, f'{tool_model_path}/net_best.pth')
    for dataset_name in ['train', 'valid', 'test']:
        synthetic_dataset_path = os.path.join(synthetic_dataset_root, dataset_name)
        dy_dataset_path = os.path.join(dy_dataset_root, dataset_name)

        dy_dataset_size = len(glob.glob(os.path.join(dy_dataset_path, '*')))
        print(f"Rolling out on the {dataset_name} set:")
        for idx in tqdm(range(dy_dataset_size)):
            synthetic_vid_path = os.path.join(synthetic_dataset_path, str(idx).zfill(3))
            if os.path.exists(synthetic_vid_path) and len(glob.glob(os.path.join(synthetic_vid_path, '*'))) > 1:
                continue

            os.system('mkdir -p ' + synthetic_vid_path)
            dy_vid_path = os.path.join(dy_dataset_path, str(idx).zfill(3))
            if not os.path.exists(dy_vid_path): 
                continue

            # load data
            state_seq = []
            frame_list = sorted(glob.glob(os.path.join(dy_vid_path, '*.h5')))
            for step in range(len(frame_list)):
                frame_name = str(step).zfill(3) + '.h5'
                data = load_data(tool_args.data_names, os.path.join(dy_vid_path, frame_name))
                state_seq.append(data[0])

            state_seq = np.stack(state_seq)

            # init_pose_seq: (n_moves, n_shapes, 3)
            # act_seq: (n_moves, n_steps, 6 * n_tools)
            # init_pose_seq = np.expand_dims(copy.deepcopy(state_seq[tool_args.n_his - 1, 
            #     tool_args.n_particles + tool_args.floor_dim:]), 0)
            # act_seq = get_act_seq_from_state_seq(tool_args, state_seq)

            # with torch.no_grad():
            #     state_pred_seq, _, _ = gnn.rollout(
            #         copy.deepcopy(state_seq[:tool_args.n_his]), np.expand_dims(init_pose_seq, 0), np.expand_dims(act_seq, 0))

            # state_pred_seq = add_shape_to_seq(tool_args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq)
            # state_pred_seq = np.concatenate((copy.deepcopy(state_seq[:tool_args.n_his]), state_pred_seq))

            # shape_quats = np.zeros((sum(tool_args.tool_dim[tool_args.env]) + tool_args.floor_dim, 4), dtype=np.float32)

            # for i in range(state_pred_seq.shape[0]):
            #     h5_data = [state_pred_seq[i], shape_quats, tool_args.scene_params]
            #     store_data(tool_args.data_names, h5_data, os.path.join(synthetic_vid_path, str(i).zfill(3) + '.h5'))

            # render_anim(tool_args, [f'GNN', 'GT'], [state_pred_seq, state_seq], res='low', 
            #         path=os.path.join(synthetic_vid_path, f'repr.mp4'))
            
            for t in range(0, state_seq.shape[0] - tool_args.time_step):
                synthetic_vid_ep_path = os.path.join(synthetic_vid_path, str(t).zfill(3))
                os.system('mkdir -p ' + synthetic_vid_ep_path)
                init_pose_seq = np.expand_dims(copy.deepcopy(state_seq[t, tool_args.n_particles + tool_args.floor_dim:, :3]), 0)
                act_seq_dense = get_act_seq_from_state_seq(tool_args, state_seq[t:, :, :3])
                act_seq_sparse = get_act_seq_from_state_seq(tool_args, copy.deepcopy(state_seq[t::tool_args.time_step, :, :3]))

                with torch.no_grad():
                    state_pred_seq, _, _ = gnn.rollout(
                        copy.deepcopy(state_seq[t]), np.expand_dims(init_pose_seq, 0), np.expand_dims(act_seq_dense, 0))

                state_pred_seq = add_shape_to_seq(tool_args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq_sparse)
                state_pred_seq = np.concatenate((copy.deepcopy(state_seq[t:t+1, :, :3]), state_pred_seq))

                shape_quats = np.zeros((sum(tool_args.tool_dim[tool_args.env]) + tool_args.floor_dim, 4), dtype=np.float32)
                for i in range(state_pred_seq.shape[0]):
                    if 'normal' in tool_type:
                        state_normals = get_normals_from_state(tool_args, state_pred_seq[i], visualize=False)
                        state_pred_seq[i] = np.concatenate([state_pred_seq[i], state_normals], axis=1)
                    h5_data = [state_pred_seq[i], shape_quats, tool_args.scene_params]
                    store_data(tool_args.data_names, h5_data, os.path.join(synthetic_vid_ep_path, str(t+i).zfill(3) + '.h5'))

                if t % 20 == 0:
                    render_anim(tool_args, [f'GNN', 'GT'], [state_pred_seq, state_seq[t::tool_args.time_step]], res='low', 
                        path=os.path.join(synthetic_vid_ep_path, f'repr.mp4'))


def main():
    args = gen_args()

    # with open('config/tool_model_map.yml', 'r') as f:
    #     tool_model_dict = yaml.load(f, Loader=yaml.FullLoader)

    # for tool_name, tool_model_path_prefix in tool_model_dict.items():
    #     tool_model_path = os.path.join(cd, '..', '..', 'models', tool_model_path_prefix)
    #     build_synthetic_dataset(args, tool_name, tool_model_path)

    dy_root = 'dump/dynamics/dump'
    suffix = 'robot_v4_surf_nocorr_full_normal'
    tool_model_dict = {
        'gripper_sym_rod': f'{dy_root}_gripper_sym_rod_{suffix}/' + \
            'dy_gt_nr=0.01_tnr=0.003_0.003_his=1_seq=3_time_step=5_chamfer_emd_0.2_0.8_rm=1_valid_Oct-13-23:42:29',
        'gripper_asym': f'{dy_root}_gripper_asym_{suffix}/' + \
            'dy_gt_nr=0.01_tnr=0.008_0.008_his=1_seq=3_time_step=5_chamfer_emd_0.2_0.8_rm=1_valid_Oct-13-23:42:29',
        'gripper_sym_plane': f'{dy_root}_gripper_sym_plane_{suffix}/' + \
            'dy_gt_nr=0.01_tnr=0.004_0.004_his=1_seq=3_time_step=5_chamfer_emd_0.2_0.8_rm=1_valid_Oct-13-23:48:13',
        'press_square': f'{dy_root}_press_square_{suffix}/' + \
            'dy_gt_nr=0.01_tnr=0.003_his=1_seq=3_time_step=5_chamfer_emd_0.2_0.8_rm=1_valid_Oct-13-23:46:13',
        'punch_square': f'{dy_root}_punch_square_{suffix}/' + \
            'dy_gt_nr=0.01_tnr=0.003_his=1_seq=3_time_step=5_chamfer_emd_0.2_0.8_rm=1_valid_Oct-13-23:46:13',
    }

    for tool_name, tool_model_path in tool_model_dict.items():
        build_synthetic_dataset(args, tool_name, tool_model_path)


if __name__ == '__main__':
    main()
