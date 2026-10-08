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

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --image_folder /local_data/jz4725/metacam/data_v3/28_metfloor/images \
#     --save_path /local_data/jz4725/VGGT-SLAM/results/28_metfloor \
#     --use_sim3

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/rebuttal/pantry1/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/pantry1 \
    --submap_size 48

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/rebuttal/pantry2/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/pantry2 \
    --submap_size 48

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/rebuttal/pantry_2traverse/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/pantry_2traverse \
    --submap_size 48

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/2ndfloor_first150/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/2ndfloor_first150 \
    --submap_size 48

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/12thfloor_first150/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/12thfloor_first150 \
    --submap_size 48

CUDA_VISIBLE_DEVICES=5 python main.py \
    --image_folder /local_data/jz4725/metacam/data_v3/28_metfloor_first150/images \
    --save_path /local_data/jz4725/VGGT-SLAM/rebuttal_v1_results/28_metfloor_first150 \
    --submap_size 48
