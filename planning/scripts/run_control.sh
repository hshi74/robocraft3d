tool_type="gripper_sym_rod_robot_v4_surf_nocorr_full"
debug=0
tool_model_name="default"
active_tool_list="default"
target_shape_name="multi_tool/dumpling_gnn"
optim_algo="RS"
CEM_sample_size=20
control_loss_type="chamfer"
subtarget=0
close_loop=1
cls_type='pcd'
planner_type='learned'

bash ./scripts/control/control.sh $tool_type $debug $tool_model_name $active_tool_list $target_shape_name $optim_algo $CEM_sample_size $control_loss_type $subtarget $close_loop $cls_type $planner_type
