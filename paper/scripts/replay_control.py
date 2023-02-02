import glob
from matplotlib.pyplot import xticks
import numpy as np
import os
import pickle
import seaborn as sns

from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


def plot_control_loss(loss_list, time_dict, path=''):
    sns.set_theme(context='paper', style='darkgrid', font_scale=8)
    plt.figure(figsize=(16, 12))
    ax = sns.lineplot(data=loss_list, linewidth=16)

    time_list_all = [len(loss_list) - 1]
    for tool_name, time_list in time_dict.items():
        for i in range(len(time_list)):
            if time_list[i] == 0: continue
            # plt.annotate(tool_name, xy=(time, loss_list[time]), 
            #     xytext=(30, 10), xycoords="figure points")
            if i == 0:
                plt.axvline(x=time_list[i], color='r', ls='--', lw=8)
            else:
                plt.axvline(x=time_list[i], color='gray', ls='-.', lw=4)

        time_list_all.extend(time_list)

    plt.xlabel('Time Steps', labelpad=20, fontsize=90)
    plt.ylabel('CD', labelpad=20, fontsize=90)
    ax.set_yticklabels([])
    ax.set(xticks=time_list_all)

    plt.tight_layout()

    # plt.figure(figsize=[16, 9])

    # time_list = list(range(len(loss_list)))
    # plt.plot(time_list, loss_list, linewidth=6)

    # for i in :
    #     plt.annotate(round(loss_list[i], 6), xy=(time_list[i], loss_list[i]), 
    #         xytext=(-30, 10), textcoords="offset points")

    # plt.xlabel('Time', fontsize=30)
    # plt.ylabel('Chamfer Distance', fontsize=30)
    # # plt.title('Training Loss', fontsize=35)
    # # plt.legend(fontsize=30)
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def main():
    rollout_root = '/scr/hshi74/projects/robocook/dump/control/control_gripper_sym_rod_robot_v4_surf_nocorr_full/alphabet_gnn'
    rollout_path_dict = {
        'R': 'control_close_max=2_RS_chamfer_Nov-13-14:11:26',
        'O': 'control_close_max=2_RS_chamfer_Nov-13-15:13:57',
        'B': 'control_close_max=2_RS_chamfer_Nov-13-16:11:00',
        'C': 'control_close_max=2_RS_chamfer_Nov-13-18:07:43',
        'K': 'control_close_max=2_RS_chamfer_Nov-13-19:10:10',
    }

    for key, value in rollout_path_dict.items():
        state_seq_dict = {}
        rollout_path = os.path.join(rollout_root, key, value)
        anim_list_path = os.path.join(rollout_path, f'anim_list.txt')
        pickle_path_list = []
        with open(anim_list_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                anim_path = lines[i].replace('file \'', '').replace('\'\n', '')
                tool_name = os.path.basename(anim_path).split('_anim')[0]
                anim_args_path_base = os.path.dirname(anim_path).replace('/anim', '/anim_args')

                for pickle_path in sorted(glob.glob(os.path.join(anim_args_path_base, '*.pkl'))):
                    if tool_name in os.path.basename(pickle_path) and not pickle_path in pickle_path_list:
                        pickle_path_list.append(pickle_path)

        for pickle_path in pickle_path_list:
            with open(pickle_path, 'rb') as f:
                args_dict = pickle.load(f)
                tool_name = os.path.basename(pickle_path).split('_anim')[0]
                if tool_name in state_seq_dict:
                    state_seq_dict[tool_name].append(args_dict['state_seqs'])
                else:
                    state_seq_dict[tool_name] = [args_dict['state_seqs']]

                if i == len(lines) - 1:
                    target = args_dict['target']

        loss_list = []
        time_now = 0
        time_dict = {}
        state_goal_norm = target - np.mean(target, axis=0)
        for tool_name, state_seq_list_list in state_seq_dict.items():
            for state_seq_list in state_seq_list_list:
                if tool_name in time_dict:
                    time_dict[tool_name].append(time_now)
                else:
                    time_dict[tool_name] = [time_now]

                for state_seq in state_seq_list:
                    time_now += state_seq.shape[0]
                    for state in state_seq:
                        state_cur = state[:target.shape[0]]
                        state_cur_norm = state_cur - np.mean(state_cur, axis=0)
                        dist_final = chamfer(state_cur_norm, state_goal_norm, pkg='numpy')
                        loss_list.append(dist_final)

        # import pdb; pdb.set_trace()
        plot_control_loss(loss_list, time_dict, 
            path=f'/scr/hshi74/projects/robocook/paper/images/{key}-{value}.png')


if __name__ == '__main__':
    main()
