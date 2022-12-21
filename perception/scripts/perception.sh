#!/usr/bin/env bash

kernprof -l perception/sample.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v4 \
	--n_particles 300 \
	--surface_sample 0 \
	--correspondance 0
