import glob
import numpy as np
import os
import sys

def main():
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", "gt")
    tast_list = ["gripper_sym_robot_v1.2_surf", "gripper_asym_robot_v1.2_surf", 
        "press_square_robot_v1.2_surf", "roller_small_robot_v1.2_surf"]
    
    dataset_list = ["train", "valid", "test"]
    for dataset in dataset_list:
        dest_dir = os.path.join(data_root_dir, 'data_all_robot_v1_surf', dataset)
        os.system(f"mkdir -p {dest_dir}")
        dp_ind = 0
        dp_dict = {}
        for tool in tast_list:
            dp_dict[tool] = []
            tool_dir = os.path.join(data_root_dir, f"data_{tool}")
            if not os.path.isdir(tool_dir): continue
            dp_dir_list = sorted(glob.glob(os.path.join(tool_dir, dataset, '*')))
            for dp in dp_dir_list:
                dp_dict[tool].append(dp_ind)
                os.system(f"ln -s {dp} {dest_dir}/{str(dp_ind).zfill(3)}")
                dp_ind += 1
        np.save(os.path.join(dest_dir, 'tool_info.npy'), dp_dict)

if __name__ == "__main__":
    main()
