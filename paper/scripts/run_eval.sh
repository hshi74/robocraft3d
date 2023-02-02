#!/usr/bin/env bash

tool_type="gripper_sym_rod_v2_surf_nocorr_full_keyframe=12"
dy_model_path="/scr/hshi74/projects/robocraft3d/dump/dynamics/dump_gripper_sym_rod_robot_v2_surf_nocorr_full_keyframe=12/dy_keyframe_nr=0.02_tnr=0.03_0.03_his=1_seq=2_time_step=1_chamfer_emd_0.5_0.5_valid_Jan-24-22:00:53/net_best.pth"
n_rollout=1

bash ./paper/scripts/eval.sh $tool_type $dy_model_path $n_rollout
