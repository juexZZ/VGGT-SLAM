# VGGT SLAM + Semantic

0. get semantic embedding by using SAM and CLIP
```bash
cd ../sam2/
source .venv/bin/activate
cd ../VGGT-SLAM/vggt_slam
python semantic_embedder.py \
  --image_folder /path/to/images \
  --output_folder /path/to/folder_for_samclip_embedding_per_image \
  --ext .png \
  --overwrite
```

1. run VGGT SLAM and build voxels in the same time
```bash
python main.py \
  --image_folder ../metacam/8thfloor/8thfloor_small_static0/images \
  --semantic_emb_dir samclip/8thfloor_small_static0/ \
  --get_voxel \
  --voxel_save_dir metacam_result/8thfloor_small_static0_scaled/ \
  --save_pointcloud metacam_result/8thfloor_small_static0_scaled/
```
But if you don't want the voxel to be built, just run the original VGGT SLAM
```bash
python main.py \
  --image_folder ../metacam/8thfloor_v2/8thfloor_static1
```

2. [optinal] visualize the SAVED voxels and the point clouds together
To visualize the voxel side-by-side with the point cloud:
```bash
python visualize_results.py \
  --pcd_path /path/to/result.pcd \
  --voxel_dir /path/to/voxel_dir \
  --voxel_render_mode cubes \
  --side_by_side
```
To visualize the voxels and the point clouds in two separate servers
```bash
python visualize_results.py \
  --pcd_path /path/to/result.pcd \
  --port 8080 \
  --voxel_npz /path/to/semantic_voxels.npz \
  --voxel_port 8081
```

3. query the voxel with text prompt, visualize the voxels and retrieve the images
```bash
python vggt_slam/query_voxelmap.py \
  --voxel_map_dir metacam_result/8thfloor_small_static0/ \
  --query_prompt "carrot" \
  --output_dir query_voxel_results \
  --image_dir ../metacam/8thfloor/8thfloor_small_static0/images
```

---

## MetaCam: undistort raw fisheye images (left/right)

This repo includes a small helper script that **only** undistorts MetaCam images (ignores extrinsics).

```bash
python3 scripts/undistort_metacam_image.py \
  --input_dir /path/to/metacam/images_or_folder \
  --output_dir /path/to/output/undistorted \
  --out_size 1600 \
  --fov_deg 90
```

Input folder can be either:
- `.../left/*.jpg` and `.../right/*.jpg`, or
- `.../left_*.jpg` and `.../right_*.jpg` in a single folder.

---

## 📚 Table of Contents
* [💻 Installation](#installation-of-vGGT-sLAM)
* [🚀 Quick Start](#quick-start)
* [📊 Running Evaluations](#running-evaluations)
* [📄 News and Updates](#News-and-Updates)
* [📄 Paper Citation](#citation)

---

## Installation of VGGT-SLAM

<!-- We created a Hugging Face Space [here]() where you can quickly run an example of VGGT-SLAM. To install 
VGGT-SLAM locally, which also enables a more detailed visualization built with Viser, use the following instructions.  -->

Make sure the following dependencies are installed before building the project:

```
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ unzip
```

Then, clone VGGT-SLAM:

```
git clone https://github.com/MIT-SPARK/VGGT-SLAM
```

```
cd VGGT-SLAM
```

### Create and activate a new conda environment

```
conda create -n vggt-slam python=3.11
```

```
conda activate vggt-slam
```

### Make the setup script executable and run it
This step will automatically download all 3rd party packages including VGGT. More details on the license for VGGT can be found [here](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).

```
./setup.sh
```

---

## Quick Start

run `python main.py --image_folder /path/to/image/folder --max_loops 1 --vis_map` replacing the image path with your folder of images. 
This will create a visualization in viser which shows the incremental construction of the map.

As an example, we provide a folder of test images in `office_loop.zip` which will generate the following map. Using the default parameters will
result in a single loop closure towards the end of the trajectory. Unzip the folder and set its path as the arguments for `--image_folder`, e.g.,

```
unzip office_loop.zip
```

and then run the below command:

```
python3 main.py --image_folder office_loop --max_loops 1 --vis_map
```

<p align="center">
  <img src="assets/office-loop-figure" width="300">
</p>

*Running in the default SL(4) mode on this folder will show significant drift 
in the projective degrees of freedom before the loop closure, and the drift will be corrected after the loop closure. You may notice drift in other scenes as well if the system goes too long without a loop closure. We are actively working on an upgraded 
VGGT-SLAM that will have significantly reduced drift and other major updates so stay tuned!*

### Collecting Custom Data

To quickly collect a test on a custom dataset, you can record a trajectory with a cell phone and convert the MOV file to a folder of images with:

```
mkdir <desired_location>/img_folder
```

And then, run the command below:

```
ffmpeg -i /path/to/video.MOV -vf "fps=10" <desired_location>/img_folder/frame_%04d.jpg
```

### Adjusting Parameters

See main.py or run `--help` from main.py to view all parameters. 
We use SL(4) mode by default, and Sim(3) mode can be enabled with `--use_sim3`. Sim(3) mode will generally have less drift than SL(4) but will not always 
be sufficient for alignment (see paper for in depth discussion on the advantages of SL(4)).

---

## Running Evaluations

To automatically run evaluation on TUM and 7-Scenes datasets, first install the datasets using the provided download instructions from [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file#examples). Set the download location of MASt3R-SLAM by setting *abs_dir* in the bash scripts 
*/evals/eval_tum.sh* and */evals/eval_7scenes.sh*

#### In Tum Dataset

To run on TUM, run `./evals/eval_tum.sh <w>` and then run `python evals/process_logs_tum.py --submap_size <w>` to analyze and print the results, where w is 
the submap size, for example:

```
./evals/eval_tum.sh 32
```

```
python evals/process_logs_tum.py --submap_size 32
```


#### In 7-Scenes Dataset

To run on 7-Scenes, run `./evals/eval_7scenes.sh <w>` and then run `python evals/process_logs_7scenes.py <w>` to analyze and print the results, where w is 
the submap size, for example:

```
./evals/eval_7scenes.sh 32
```

```
python evals/process_logs_7scenes.py 32
```

--- 

By default, ever scene will be run for 5 trials, this can be changed inside the bash scripts.

To visualize the maps as they being constructed, inside the bash scripts add `--vis_map`. This will update the viser map each time the submap is updated. 

## News and Updates

* August 2025: SL(4) optimization is integrated into the official GTSAM repo
* September 2025: Accepted to Neurips 2025
* November 2025: Featured in MIT News [article](https://news.mit.edu/2025/teaching-robots-to-map-large-environments-1105)

## Acknowledgement

This work is supported in part by the NSF Graduate Research Fellowship Program under Grant
2141064, the ONR RAPID program, and the National Research Foundation of Korea (NRF) grant
funded by the Korea government (MSIT) (No. RS-2024-00461409). The authors would like to
gratefully acknowledge Riku Murai for assisting us with benchmarking.

## Citation

If our code is helpful, please cite our paper as follows:

```
@article{maggio2025vggt-slam,
  title={VGGT-SLAM: Dense RGB SLAM Optimized on the SL (4) Manifold},
  author={Maggio, Dominic and Lim, Hyungtae and Carlone, Luca},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2025}
}
```

