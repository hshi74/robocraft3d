tool_type="gripper_sym_rod_robot_v4_surf_nocorr_full"
debug=1
target_shape_name="alphabet_gnn/K"
optim_algo="RS"
CEM_sample_size=20
control_loss_type="chamfer"
subtarget=0
close_loop=0
planner_type='learned'

bash ./planning/scripts/control.sh $tool_type $debug $target_shape_name $optim_algo $CEM_sample_size $control_loss_type $subtarget $close_loop $planner_type
