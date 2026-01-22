import os
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT


parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization. Try increasing (such as 4) to reduce lag in visualizing large maps.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")
parser.add_argument("--save_pointcloud", type=str, default=None, help="Directory to save the point cloud file (e.g., output.pcd). If provided, point cloud will be saved before visualization.")
parser.add_argument("--keep_alive", action="store_true", help="Keep the viser server alive until manual shutdown (press Enter to exit)")
parser.add_argument("--semantic_emb_dir", type=str, default=None, help="Directory containing per-image semantic embeddings as .npz (same stem as image filename), key 'embedding'=(H,W,d).")
parser.add_argument("--get_voxel", action="store_true", help="Build, save, and visualize a global semantic voxel map after SLAM finishes (requires --semantic_emb_dir).")
parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size (meters) for semantic voxelization.")
parser.add_argument("--voxel_save_dir", type=str, default=None, help="Directory to save semantic voxel map (writes voxels.npz + frame_names.json).")
parser.add_argument("--voxel_port", type=int, default=8081, help="Port for semantic voxelmap viser server.")
parser.add_argument("--voxel_point_size", type=float, default=0.01, help="Point size for voxel visualization in viser.")
parser.add_argument("--colmap_images_txt", type=str, default=None, help="Optional COLMAP images.txt to align predicted map to real-world scale (Sim3).")
parser.add_argument("--align_no_scale", action="store_true", help="If set, align with SE3 only (no scale). Default is Sim3 with scale.")

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride = args.vis_stride,
        vis_point_size = args.vis_point_size,
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            semantic_embeddings = None
            if args.semantic_emb_dir is not None:
                # Load per-image embeddings from disk and stack into (S,H,W,d).
                print(f"Loading semantic embeddings from {args.semantic_emb_dir}...")
                embs = []
                for img_path in image_names_subset:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    emb_path = os.path.join(args.semantic_emb_dir, f"{stem}.npz")
                    if not os.path.exists(emb_path):
                        raise FileNotFoundError(f"Missing semantic embedding for {img_path}: {emb_path}")
                    emb = np.load(emb_path)["embedding"]
                    embs.append(emb)
                semantic_embeddings = np.stack(embs, axis=0)
                print(f"Loaded semantic embeddings shape: {semantic_embeddings.shape}")

            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, semantic_embeddings=semantic_embeddings)

            data.append(predictions["intrinsic"][:,0,0])

            solver.add_points(predictions)

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    # Optional global alignment to COLMAP GT camera centers (sets real-world scale).
    # This should be done after pose-graph optimization and homography updates, but before saving/voxelizing.
    if args.colmap_images_txt is not None:
        print(f"Aligning map to COLMAP poses: {args.colmap_images_txt}")
        solver.map.align_scale_to_colmap(args.colmap_images_txt, with_scale=not args.align_no_scale)

    if not args.vis_map and not args.get_voxel:
        # just show the map after all submaps have been processed
        print("Updating all submap visualizations...")
        solver.update_all_submap_vis()

    # Build + (optionally) save + visualize the global semantic voxel map
    if args.get_voxel:
        print("Build and visualize semantic voxel map...")
        if args.semantic_emb_dir is None:
            raise ValueError("--get_voxel requires --semantic_emb_dir so semantic embeddings are available.")
        print(f"Building semantic voxel map with voxel size {args.voxel_size}...")
        semantic_voxel_map = solver.map.build_semantic_voxel_map(voxel_size=args.voxel_size)
        if args.voxel_save_dir is not None:
            semantic_voxel_map.save_to_directory(args.voxel_save_dir)
            print(f"Saved semantic voxel map to {args.voxel_save_dir}")
        # Visualize in a separate viser server
        print(f"Visualizing semantic voxel map on port {args.voxel_port}...")
        semantic_voxel_map.visualize(port=args.voxel_port, point_size=args.voxel_point_size,
                                     render_mode="cubes", color_mode="pca", wireframe=False, opacity=0.5)

    # Save point cloud if requested
    if args.save_pointcloud:
        print(f"Saving point cloud to {args.save_pointcloud} result.pcd...")
        os.makedirs(args.save_pointcloud, exist_ok=True)
        file_name = os.path.join(args.save_pointcloud, "result.pcd")
        solver.map.write_points_to_file(file_name)
        print("Point cloud saved successfully!")

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()

    # Keep viser server alive if visualization is shown (or if explicitly requested)
    # Visualization is shown if --vis_map is set or if final map is displayed (when not using --vis_map)
    visualization_shown = args.vis_map or (not args.vis_map and solver.map.get_num_submaps() > 0)
    should_keep_alive = args.keep_alive or visualization_shown
    if should_keep_alive and not solver.gradio_mode:
        print("\nViser server is running. Press Enter to exit...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
