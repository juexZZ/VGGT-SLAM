#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python semantic_embedder.py \
  --image_folder /local_data/jz4725/metacam/data_v3/8thfloor_add/images \
  --output_folder /local_data/jz4725/metacam/data_v3/8thfloor_add/semantic_emb \
  --ext .png

CUDA_VISIBLE_DEVICES=1 python semantic_embedder.py \
  --image_folder /local_data/jz4725/metacam/data_v3/8thfloor_remove/images \
  --output_folder /local_data/jz4725/metacam/data_v3/8thfloor_remove/semantic_emb \
  --ext .png

CUDA_VISIBLE_DEVICES=1 python semantic_embedder.py \
  --image_folder /local_data/jz4725/metacam/data_v3/8thfloor_move/images \
  --output_folder /local_data/jz4725/metacam/data_v3/8thfloor_move/semantic_emb \
  --ext .png