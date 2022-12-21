tool_type="gripper_sym_rod_robot_v4_surf_nocorr_full_normal_keyframe=16"
data_type="keyframe"
debug=1
loss_type="chamfer_emd"
n_his=1
sequence_length=2
time_step=3
rigid_motion=1
attn=0
chamfer_weight=0.5
emd_weight=0.5
neighbor_radius=0.01
tool_neighbor_radius="default"
train_set_ratio=1.0

bash ./scripts/dynamics/train.sh $tool_type $data_type $debug $loss_type $n_his $sequence_length $time_step $rigid_motion $attn $chamfer_weight $emd_weight $neighbor_radius $tool_neighbor_radius $train_set_ratio
