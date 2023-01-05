#!/usr/bin/env bash

python dynamics/scripts/build_dataset_keyframe.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v1_surf_nocorr_full_normal \
	--debug 0
