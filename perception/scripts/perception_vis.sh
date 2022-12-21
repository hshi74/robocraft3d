#!/usr/bin/env bash

kernprof -l perception/sample_vis.py \
	--stage perception \
	--tool_type roller_large_robot_v4 \
	--n_particles 300 \
	--surface_sample 1 \
	--correspondance 0
