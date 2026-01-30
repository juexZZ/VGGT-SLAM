#!/bin/bash

# slam code
CUDA_VISIBLE_DEVICES=2 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/2ndfloor/images \
    --save_path /local_data/jz4725/VGGT-SLAM/results/2ndfloor \
    --use_sim3

CUDA_VISIBLE_DEVICES=2 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/12thfloor/images \
    --save_path /local_data/jz4725/VGGT-SLAM/results/12thfloor \
    --use_sim3