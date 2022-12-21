import glob
import numpy as np
import open3d as o3d
import os
import pickle
import readchar
import subprocess
import sys
import torch
import yaml

torch.set_printoptions(sci_mode=False)

from control_utils import *
from datetime import datetime
from std_msgs.msg import UInt8
from tool import *
from config.config import gen_args
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


args = gen_args()

if args.close_loop:
    import rospy
    from perception.sample_pcd import ros_bag_to_pcd
    from rospy.numpy_msg import numpy_msg
    from rospy_tutorials.msg import Floats
    from std_msgs.msg import UInt8, String


command_feedback = 0
def command_fb_callback(msg):
    global command_feedback
    if msg.data > 0:
        command_feedback = msg.data


def get_test_name(args):
    test_name = ['control']
    if args.close_loop:
        test_name.append('close')
    else:
        test_name.append('open')

    if len(args.active_tool_list) == 1:
        if 'rm=1' in args.tool_model_dict[args.tool_type]:
            test_name.append('rm')
        if 'attn=1' in args.tool_model_dict[args.tool_type]:
            test_name.append('attn')

    test_name.append(f'max={args.max_n_actions}')
    test_name.append(args.optim_algo)
    if 'CEM' in args.optim_algo and not args.debug:
        test_name.append(f'{args.CEM_sample_size}')
        test_name.append(f'{args.CEM_decay_factor}')
    test_name.append(args.control_loss_type)
    
    if args.debug: test_name.append('debug')

    test_name.append(datetime.now().strftime("%b-%d-%H:%M:%S"))

    return '_'.join(test_name)


cd = os.path.dirname(os.path.realpath(sys.argv[0]))

rollout_root = os.path.join(cd, '..', 'dump', 'control', f'control_{args.tool_type}', 
    args.target_shape_name, get_test_name(args))
os.system('mkdir -p ' + rollout_root)

for dir in ['states', 'raw_data']:
    os.system('mkdir -p ' + os.path.join(rollout_root, dir))


class MPController(object):
    def __init__(self):
        self.get_target_shapes()
        self.load_tool()


    def get_target_shapes(self):
        target = os.path.basename(args.target_shape_name)
        target_dir = os.path.join(cd, '..', 'target_shapes', args.target_shape_name)

        if 'sim' in args.target_shape_name:
            prefix = target
        else:
            prefix = f"{target}/{target.split('_')[0]}"

        target_shape = {}
        for type in ['sparse', 'dense', 'surf']:
            if type == 'sparse':
                target_frame_path = os.path.join(target_dir, f'{prefix}.h5')
            else:
                target_frame_path = os.path.join(target_dir, f'{prefix}_{type}.h5')

            if os.path.exists(target_frame_path):
                target_data = load_data(args.data_names, target_frame_path)
                target_shape[type] = target_data[0]

        self.target_shape = target_shape


    def load_tool(self, tool_name='gripper_sym_rod'):
        with open('config/plan_params.yml', 'r') as f:
            plan_params = yaml.load(f, Loader=yaml.FullLoader)

        with open('config/model_map.yml', 'r') as f:
            model_dict = yaml.load(f, Loader=yaml.FullLoader)

        if 'sim' in args.planner_type:
            model_path_list = None
        else:
            tool_model_names = model_dict[args.planner_type][tool_name]
            if isinstance(tool_model_names, list):
                model_path_list = []
                for tool_model_name in tool_model_names:
                    model_path_list.append(os.path.join(cd, '..', 'models', args.planner_type, tool_model_name))
            else:
                model_path_list = [os.path.join(cd, '..', 'models', args.planner_type, tool_model_names)]
        self.tool = Tool(args, tool_name, args.planner_type, plan_params[tool_name], model_path_list)


    def get_state_from_ros(self, ros_data_path):
        command_time = datetime.now().strftime("%b-%d-%H:%M:%S")
        ros_pcd_path_prefix = os.path.join(ros_data_path, command_time)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.command_pub.publish((String(f'{command_time}.shoot.pcd')))
            
            self.ros_data_path_pub.publish(String(ros_pcd_path_prefix))
            if os.path.exists(ros_pcd_path_prefix + '.bag'):
                print(f"[INFO] Received data from cameras!")
                break

            rate.sleep()

        pcd_dense, pcd_sparse, state_cur = ros_bag_to_pcd(args, ros_pcd_path_prefix + '.bag', visualize=False)
        
        state_cur_dict = {
            'tensor': torch.tensor(state_cur[:args.n_particles], 
                device=args.device, dtype=torch.float32).unsqueeze(0),
            'dense': torch.tensor(np.asarray(pcd_dense.points), 
                device=args.device, dtype=torch.float32).unsqueeze(0),
        }

        return state_cur_dict


    def control(self):
        if args.close_loop:
            ros_data_path = os.path.join(rollout_root, 'raw_data')
            # args.env = 'hook'
            state_init_dict = self.get_state_from_ros(args.env, ros_data_path)
        else:
            target_dir = os.path.join(cd, '..', 'target_shapes', args.target_shape_name)

            if 'sim' in args.target_shape_name:
                name = 'start'
            else:
                name = '000'

            init_state_path = os.path.join(target_dir, name, f'{name}.h5')
            dense_init_state_path = os.path.join(target_dir, name, f'{name}_dense.h5')
            surf_init_state_path = os.path.join(target_dir, name, f'{name}_surf.h5')

            init_state_data = load_data(args.data_names, init_state_path)
            dense_init_state_data = load_data(args.data_names, dense_init_state_path)
            surf_init_state_data = load_data(args.data_names, surf_init_state_path)

            init_state = {
                'sparse': init_state_data[0], 
                'dense': dense_init_state_data[0],
                'surf': surf_init_state_data[0]
            }

            state_init_dict = {
                'tensor': torch.tensor(init_state['surf'][:args.n_particles], 
                    device=args.device, dtype=torch.float32).unsqueeze(0),
                'dense': torch.tensor(init_state['dense'][:-args.floor_dim],
                    device=args.device, dtype=torch.float32).unsqueeze(0),
            }


        for dir in ['param_seqs', 'anim', 'anim_args', 'states', 'optim_plots', 'raw_data']:
            os.system('mkdir -p ' + os.path.join(rollout_root, dir))

        # plan the actions given the tool
        param_seq, state_seq, info_dict, state_cur_dict = self.plan(state_init_dict, rollout_root)

        self.summary([param_seq, state_seq, info_dict, state_cur_dict])


    def execute(self, param_seq):
        print(f"Executing...")
        self.param_seq_pub.publish(param_seq.flatten().astype(np.float32))
        command_time = datetime.now().strftime("%b-%d-%H:%M:%S")
        self.command_pub.publish(String(f'{command_time}.run'))


    def plan(self, state_cur_dict, rollout_path, max_n_actions=5, pred_err_bar=0.02):
        global command_feedback

        param_seq, state_seq, info_dict = self.tool.rollout(
            state_cur_dict, self.target_shape, rollout_path, args.max_n_actions
        )

        if args.close_loop:
            act_len = state_seq.shape[0] // param_seq.shape[0]
            act_start = 0
            act_end = 1
            param_seq_todo = param_seq[act_start:act_end].numpy()
            param_seq_pred = param_seq_todo
            state_seq_pred = state_seq[:act_len][:args.n_particles]
            state_pred_tensor = torch.tensor(state_seq_pred[-1], device=args.device, 
                dtype=torch.float32).unsqueeze(0)
            info_dict_pred = info_dict

            ros_data_path = os.path.join(rollout_path, 'raw_data')
            # loss_dict = {'Chamfer': [], 'EMD': [], 'IOU': []}
            while not rospy.is_shutdown():
                self.execute(param_seq_todo)

                while command_feedback != 1:
                    continue

                command_feedback = 0
                
                print('Waiting for disturbance... Press enter when you finish...')
                readchar.readkey()
                
                state_cur_dict = self.get_state_from_ros(ros_data_path)

                pred_err = chamfer(state_cur_dict['tensor'].squeeze(), state_pred_tensor.squeeze(), pkg='torch')
                print(f"The prediction error is {pred_err}!")
                # chamfer_loss, emd_loss = self.eval_state(state_cur_dict['tensor'].cpu(), 
                #     step, best_target_idx, state_pred=state_pred_tensor.cpu(), pred_err=pred_err)
            
                # UP TO HERE

                if not best_tool_name in args.precoded_tool_list and param_seq_pred.shape[0] < max_n_actions:
                    # TODO: Need to tune this number
                    if pred_err > 0 and pred_err < pred_err_bar:
                        print(f"The prediction is good enough!")
                        if act_end < param_seq.shape[0]:
                            # move to the next action
                            act_start = act_end
                        elif 'roller' in best_tool_name:
                            param_seq, state_seq, info_dict = self.active_tool_dict[best_tool_name].rollout(
                                state_cur_dict, self.target_shapes[best_target_idx], rollout_path, 
                                min(max_n_actions - param_seq_pred.shape[0], args.max_n_actions)
                            )
                            act_start = 0
                        else:
                            break
                    else:
                        # figure out a new solution
                        param_seq, state_seq, info_dict = self.active_tool_dict[best_tool_name].rollout(
                            state_cur_dict, self.target_shapes[best_target_idx], rollout_path, 
                            min(max_n_actions - param_seq_pred.shape[0], args.max_n_actions)
                        )
                        act_start = 0

                    act_end = act_start + 1

                    param_seq_todo = param_seq[act_start:act_end].numpy()
                    param_seq_pred = np.concatenate((param_seq_pred, param_seq_todo))
                    state_seq_pred = np.concatenate((state_seq_pred, state_seq[act_start*act_len:act_end*act_len]))
                    state_pred_tensor = torch.tensor(state_seq_pred[-1], device=args.device, 
                        dtype=torch.float32).unsqueeze(0)

                    for key, value in info_dict.items():
                        info_dict_pred[key].extend(value)
                else:
                    break

            best_param_seq = param_seq_pred 
            best_state_seq = state_seq_pred 
            best_info_dict = info_dict_pred
        else:
            state_cur_dict['tensor'] = torch.tensor(best_state_seq[-1][:args.n_particles], 
                device=args.device, dtype=torch.float32).unsqueeze(0)
            # state_cur_dict['tensor'] = torch.tensor(self.target_shapes[step]['surf'], 
            #     device=args.device, dtype=torch.float32).unsqueeze(0)
            state_cur_dict['images'] = self.target_shapes[step]['images']
            state_cur_dict['raw_pcd'] = self.target_shapes[step]['raw_pcd']

        return best_tool_name, best_param_seq, best_state_seq, best_info_dict, state_cur_dict


    # def eval_state(self, state_cur, step, target_idx, state_pred=None, pred_err=0):
    #     target_idx = min(target_idx, len(self.target_shapes) - 1)
    #     if args.surface_sample:
    #         state_goal = torch.tensor(self.target_shapes[target_idx]['surf'], dtype=torch.float32).unsqueeze(0)
    #     else:
    #         state_goal = torch.tensor(self.target_shapes[target_idx]['sparse'], dtype=torch.float32).unsqueeze(0)

    #     state_cur_norm, state_goal_norm = normalize_state(args, state_cur, state_goal, pkg='torch')

    #     chamfer_loss = chamfer(state_cur_norm.squeeze(0), state_goal_norm.squeeze(0), pkg='torch')
    #     emd_loss = emd(state_cur_norm.squeeze(0), state_goal_norm.squeeze(0), pkg='torch')

    #     # state_cur_upsample = upsample(state_cur[0], visualize=False)
    #     # iou_loss = 1 - iou(state_cur_upsample, self.target_shapes[target_idx]['dense'], voxel_size=0.003, visualize=False)
    #     # iou_loss = torch.tensor([iou_loss], dtype=torch.float32)

    #     if state_pred is not None:
    #         state_pred_norm = (state_pred - torch.mean(state_pred, dim=1)) / torch.std(state_pred, dim=1)
    #         render_frames(args, [f'State', f'State Pred={round(pred_err.item(), 6)}', 'State Normalized', 'State Pred Normalized'], 
    #             [state_cur, state_pred, state_cur_norm, state_pred_norm], 
    #             axis_off=False, focus=[True, True, False, False], 
    #             target=[state_goal, state_goal, state_goal_norm, state_goal_norm], 
    #             path=os.path.join(rollout_root, 'states'), 
    #             name=f"state_{step}.png")
    #     else:
    #         render_frames(args, [f'State', 'State Normalized'], 
    #             [state_cur, state_cur_norm], axis_off=False, focus=[True, False], 
    #             target=[state_goal, state_goal_norm], path=os.path.join(rollout_root, 'states'), 
    #             name=f"state_{step}.png")

    #     return chamfer_loss.item(), emd_loss.item() # , iou_loss.item()


    def summary(self, data):
        tool_list, loss_dict, param_seq_dict, state_seq_list, info_dict_list = data

        print(f"{'#'*27} MPC SUMMARY {'#'*28}")
        for key, value in param_seq_dict.items():
            print(f"{key}: {value}")
        
        if args.close_loop:
            state_cur_dict = self.get_state_from_ros(tool_list[-1], os.path.join(rollout_root, 'raw_data'))
            state_cur = state_cur_dict['tensor'].squeeze().cpu().numpy()
        else:
            state_cur = state_seq_list[-1][-1, :args.n_particles]

        state_cur_norm = state_cur - np.mean(state_cur, axis=0)
        state_goal = self.target_shapes[-1]['surf']
        state_goal_norm = state_goal - np.mean(state_goal, axis=0)
        dist_final = chamfer(state_cur_norm, state_goal_norm, pkg='numpy')
        print(f'FINAL chamfer distance: {dist_final}')

        with open(os.path.join(rollout_root, 'planning_time.txt'), 'r') as f:
            print(f'TOTAL planning time (s): {f.read()}')

        with open(os.path.join(rollout_root, f'MPC_param_seq.yml'), 'w') as f:
            yaml.dump(param_seq_dict, f, default_flow_style=True)

        for info_dict in info_dict_list:
            for p in info_dict['subprocess']:
                p.communicate()

        anim_list_path = os.path.join(rollout_root, f'anim_list.txt')
        with open(anim_list_path, 'w') as f:
            for i, tool in enumerate(tool_list):
                anim_path_list = sorted(glob.glob(os.path.join(rollout_root, str(i).zfill(3), 'anim', '*.mp4')))
                for anim_path in anim_path_list:
                    anim_name = os.path.basename(anim_path)
                    if tool in anim_name and not 'RS' in anim_name and not 'CEM' in anim_name \
                        and not 'GD' in anim_name and not 'sim' in anim_name:
                        f.write(f"file '{anim_path}'\n")

        mpc_anim_path = os.path.join(rollout_root, f'MPC_anim.mp4')
        subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', anim_list_path, '-c', 'copy', mpc_anim_path], 
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    tee = Tee(os.path.join(rollout_root, 'control.txt'), 'w')

    mpcontroller = MPController()

    if args.close_loop:
        rospy.init_node('control', anonymous=True)

        mpcontroller.param_seq_pub = rospy.Publisher('/param_seq', numpy_msg(Floats), queue_size=10)
        mpcontroller.command_pub = rospy.Publisher('/command', String, queue_size=10)
        mpcontroller.ros_data_path_pub = rospy.Publisher('/raw_data_path', String, queue_size=10)
        rospy.Subscriber('/command_feedback', UInt8, command_fb_callback)
    
    mpcontroller.control()


if __name__ == '__main__':
    main()
