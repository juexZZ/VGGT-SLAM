#!/bin/bash

# slam code
# CUDA_VISIBLE_DEVICES=2 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/2ndfloor/images \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/2ndfloor \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=2 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/12thfloor/images \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/12thfloor \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/2ndfloor/image_trav_1 \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/2ndfloor_trav_1 \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/12thfloor/image_trav_1 \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/12thfloor_trav_1 \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/6metro_add/image_trav_1 \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/6metro_add_trav_1 \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/28_metfloor/image_trav_1 \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/28_metfloor_trav_1 \
#     --use_sim3

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/28_metfloor/image_trav_12 \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/28_metfloor_trav_12 \
#     --use_sim3

CUDA_VISIBLE_DEVICES=0 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/28_metfloor/images \
    --save_path /local_data/jz4725/VGGT-SLAM/results/28_metfloor \
    --use_sim3