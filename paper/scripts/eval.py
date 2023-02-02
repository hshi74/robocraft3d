import glob
import numpy as np
import os
import torch

from config.config import *
from dynamics.gnn import GNN
from perception.pcd_utils import alpha_shape_mesh_reconstruct
from tqdm import tqdm
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)


def evaluate(args, load_args=False):
    # if not 'dump' in args.dy_model_path:
    #     cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    #     args.dy_model_path = os.path.join(cd, '..', 'dump', 'dynamics', 
    #         f'dump_{args.tool_type}', args.dy_model_path)

    args.dy_out_path = os.path.dirname(args.dy_model_path) 

    args_path = os.path.join(args.dy_out_path, 'args.npy')
    if load_args and os.path.exists(args_path):
        args.__dict__ = np.load(args_path, allow_pickle=True).item()

    print_args(args)

    gnn = GNN(args, args.dy_model_path)

    dataset = 'test'
    idx = 14
    vid_path = os.path.join(args.dy_data_path, dataset, str(idx).zfill(3))

    gt_vid_path = vid_path

    # load data
    n_frames = len(glob.glob(os.path.join(gt_vid_path, '*.h5')))
    state_gt_seq_dense = []
    for step in range(0, n_frames, args.data_time_step):
        frame_name = str(step).zfill(3) + '.h5'
        gt_data = load_data(args.data_names, os.path.join(gt_vid_path, frame_name))
        state_gt_seq_dense.append(gt_data[0])
    state_gt_seq_dense = np.stack(state_gt_seq_dense)
    act_seq_dense = get_act_seq_from_state_seq(args, state_gt_seq_dense[:, :, :3])

    state_gt_seq = []
    frame_start = state_gt_seq_dense.shape[0] - 1 - (state_gt_seq_dense.shape[0] - 1) // args.time_step * args.time_step
    frame_list = list(range(frame_start, state_gt_seq_dense.shape[0], args.time_step))
    # print(frame_start,  state_gt_seq_dense.shape[0], args.time_step)
    # print(frame_list)
    state_gt_seq = state_gt_seq_dense[frame_list]

    state_seq = state_gt_seq

    if args.surface_sample:
        state_surf_seq = state_seq
    else:
        surf_data_path = f'{args.dy_data_path}_surf_nocorr'
        if args.full_repr:
            surf_data_path += '_full'

        state_surf_seq = []
        for step in frame_list:
            frame_name = str(step).zfill(3) + '.h5'
            surf_data = load_data(args.data_names, os.path.join(surf_data_path, dataset, str(idx).zfill(3), frame_name))
            state_surf_seq.append(surf_data[0])
        state_surf_seq = np.stack(state_surf_seq)

    # init_pose_seq: (n_moves, n_shapes, 3)
    # act_seq: (n_moves, n_steps, 6 * n_tools)
    init_pose_seq = np.expand_dims(copy.deepcopy(state_gt_seq[0, args.n_particles + args.floor_dim:, :3]), 0)
    act_seq = get_act_seq_from_state_seq(args, state_gt_seq[:, :, :3])

    # import pdb; pdb.set_trace()
    with torch.no_grad():
        state_pred_seq, attn_mask_pred, rels_pred = gnn.rollout(
            copy.deepcopy(state_gt_seq[0]), np.expand_dims(init_pose_seq, 0), np.expand_dims(act_seq_dense, 0))

    state_pred_seq = add_shape_to_seq(args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq)
    state_pred_seq = np.concatenate((copy.deepcopy(state_gt_seq[:1, :, :3]), state_pred_seq))

    chamfer_loss_list = []
    emd_loss_list = []
    for i in range(state_seq.shape[0]):
        state_pred = state_pred_seq[i, :args.n_particles]
        target_state = state_seq[i, :args.n_particles, :3]
        chamfer_loss_list.append(chamfer(state_pred, target_state))
        emd_loss_list.append(emd(state_pred, target_state))

    print(chamfer_loss_list[-1])

    for i, loss_list in enumerate([chamfer_loss_list, emd_loss_list]):
        ax = sns.lineplot(x=list(range(len(loss_list))), y=loss_list, linewidth=5)

        ax.set(xlabel=None)
        ax.set(ylabel=None)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax
    # render_anim(gnn.args, [f'Eval', 'GT'], [state_pred_seq, state_seq[:, :, :3]], draw_set=['dough', 'tool'], axis_off=True, fps=10,
    #     path=os.path.join('paper', 'images', f'{dataset}_{str(idx).zfill(3)}.gif'))


if __name__ == '__main__':
    args = gen_args()
    dy_model_path_list = [
        "/scr/hshi74/projects/robocraft3d/dump/dynamics/dump_gripper_sym_rod_robot_v2_surf_nocorr_full_keyframe=12/dy_keyframe_nr=0.02_tnr=0.03_0.03_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_rm=1_valid_Jan-15-11:54:47/net_best.pth",
        "/scr/hshi74/projects/robocraft3d/dump/dynamics/dump_gripper_sym_rod_robot_v2_surf_nocorr_full_keyframe=12/dy_keyframe_nr=0.02_tnr=0.03_0.03_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Jan-24-22:00:53/net_best.pth"
    ]

    plt.figure(figsize=(8, 6))
    for dy_model_path in dy_model_path_list:
        print(dy_model_path)
        plt.gca().set_prop_cycle(None)
        args.dy_model_path = dy_model_path
        ax = evaluate(args, load_args=True)

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")

    plt.tight_layout()
    plt.savefig(f'paper/images/{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
