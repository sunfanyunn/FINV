#!/bin/bash -ex

scene_id=scene0038_00

# scannet chair & table
python main.py scene_id=${scene_id}

# car
#CUDA_VISIBLE_DEVICES=$device python car_main.py --gan $2 --out_root /data/sunfanyun/autorf/data/full_val_nusc_v1_0/ $3
