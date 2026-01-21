#!/usr/bin/env python3
"""
Script to visualize saved VGGT-SLAM results without rerunning the model.
Loads point clouds and optionally poses/images from saved files.
"""

import os
import glob
import argparse
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple

import viser
import viser.transforms as viser_tf
from vggt_slam.solver import Viewer


def load_point_cloud(pcd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from .pcd file.
    
    Returns:
        points: (N, 3) array of point positions
        colors: (N, 3) array of point colors (0-255 uint8)
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Convert colors to uint8 if they're in [0, 1] range
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    return points, colors


def load_poses(poses_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load poses from text file.
    Format: frame_id x y z qx qy qz qw (quaternion)
    
    Returns:
        extrinsics: (N, 3, 4) array of cam2world extrinsics
        frame_ids: List of frame ID strings
    """
    poses = []
    frame_ids = []
    
    with open(poses_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            frame_id = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Convert quaternion to rotation matrix
            quat = np.array([qw, qx, qy, qz])  # scipy uses w, x, y, z
            rot = R.from_quat(quat)
            rot_matrix = rot.as_matrix()
            
            # Build 4x4 transformation matrix (cam2world)
            T = np.eye(4)
            T[0:3, 0:3] = rot_matrix
            T[0:3, 3] = [x, y, z]
            
            # Extract 3x4 extrinsic matrix
            extrinsic = T[0:3, :]
            
            poses.append(extrinsic)
            frame_ids.append(frame_id)
    
    if len(poses) == 0:
        raise ValueError(f"No valid poses found in {poses_path}")
    
    return np.stack(poses, axis=0), frame_ids


def load_images(image_folder: str, frame_ids: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """
    Load images from folder.
    If frame_ids is provided, tries to match images to frame IDs.
    Otherwise, loads all images in sorted order.
    
    Returns:
        images: (N, 3, H, W) array of images in [0, 1] range, or None if no images found
    """
    if not os.path.exists(image_folder):
        return None
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if len(image_files) == 0:
        return None
    
    # Sort images
    image_files = sorted(image_files)
    
    # If frame_ids provided, try to match
    if frame_ids is not None and len(frame_ids) > 0:
        # Try to match by extracting numbers from filenames
        matched_images = []
        for frame_id in frame_ids:
            # Try to find image with this frame_id in filename
            found = False
            for img_file in image_files:
                filename = os.path.basename(img_file)
                # Extract number from filename
                import re
                match = re.search(r'\d+(?:\.\d+)?', filename)
                if match and abs(float(match.group()) - float(frame_id)) < 1e-6:
                    matched_images.append(img_file)
                    found = True
                    break
            if not found:
                # If no match, use first available image
                if len(image_files) > 0:
                    matched_images.append(image_files[0])
        image_files = matched_images if matched_images else image_files
    
    # Limit to number of poses if available
    if frame_ids is not None:
        image_files = image_files[:len(frame_ids)]
    
    # Load images
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Convert to (3, H, W)
        img = img.transpose(2, 0, 1)
        images.append(img)
    
    if len(images) == 0:
        return None
    
    return np.stack(images, axis=0)


def visualize_results(
    pcd_path: str,
    poses_path: Optional[str] = None,
    image_folder: Optional[str] = None,
    vis_stride: int = 1,
    vis_point_size: float = 0.003,
    port: int = 8080,
):
    """
    Visualize saved VGGT-SLAM results.
    
    Args:
        pcd_path: Path to .pcd point cloud file
        poses_path: Optional path to poses text file
        image_folder: Optional path to folder containing images
        vis_stride: Stride for point cloud visualization (1 = all points)
        vis_point_size: Size of points in visualization
        port: Port for viser server
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")
    
    print(f"Loading point cloud from {pcd_path}...")
    points, colors = load_point_cloud(pcd_path)
    print(f"Loaded {len(points)} points")
    
    # Apply stride if needed
    if vis_stride > 1:
        points = points[::vis_stride]
        colors = colors[::vis_stride]
        print(f"Downsampled to {len(points)} points (stride={vis_stride})")
    
    # Initialize viewer
    print(f"Starting viser server on port {port}...")
    viewer = Viewer(port=port)
    
    # Add point cloud
    print("Adding point cloud to visualization...")
    viewer.server.scene.add_point_cloud(
        name="pointcloud",
        points=points,
        colors=colors,
        point_size=vis_point_size,
        point_shape="circle",
    )
    
    # Load and visualize poses if provided
    if poses_path is not None:
        if not os.path.exists(poses_path):
            print(f"Warning: Poses file not found: {poses_path}. Skipping pose visualization.")
        else:
            print(f"Loading poses from {poses_path}...")
            extrinsics, frame_ids = load_poses(poses_path)
            print(f"Loaded {len(extrinsics)} poses")
            
            # Load images if folder provided
            images = None
            if image_folder is not None:
                print(f"Loading images from {image_folder}...")
                images = load_images(image_folder, frame_ids)
                if images is not None:
                    print(f"Loaded {len(images)} images")
                else:
                    print("No images found or could not load images")
            
            # Visualize poses
            if images is not None and len(images) == len(extrinsics):
                # Use images for frustums
                print("Visualizing camera poses with images...")
                viewer.visualize_frames(extrinsics, images, submap_id=0, image_scale=0.5)
            else:
                # Just show coordinate frames without images
                print("Visualizing camera poses (no images)...")
                for i, extrinsic in enumerate(extrinsics):
                    cam2world_3x4 = extrinsic
                    T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
                    
                    frame_name = f"pose_{i}"
                    frame_axis = viewer.server.scene.add_frame(
                        frame_name,
                        wxyz=T_world_camera.rotation().wxyz,
                        position=T_world_camera.translation(),
                        axes_length=0.05,
                        axes_radius=0.002,
                        origin_radius=0.002,
                    )
                    frame_axis.visible = viewer.gui_show_frames.value
    
    print("\n" + "="*60)
    print("Visualization ready!")
    print(f"Open your browser to: http://localhost:{port}")
    print("="*60)
    print("Press Enter to exit...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved VGGT-SLAM results without rerunning the model"
    )
    parser.add_argument(
        "--pcd_path",
        type=str,
        required=True,
        help="Path to .pcd point cloud file (saved with --save_pointcloud)"
    )
    parser.add_argument(
        "--poses_path",
        type=str,
        default=None,
        help="Optional path to poses text file (saved with --log_results)"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Optional path to folder containing images (for camera frustum visualization)"
    )
    parser.add_argument(
        "--vis_stride",
        type=int,
        default=1,
        help="Stride for point cloud visualization (1 = all points, higher = fewer points)"
    )
    parser.add_argument(
        "--vis_point_size",
        type=float,
        default=0.003,
        help="Size of points in visualization"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server"
    )
    
    args = parser.parse_args()
    
    visualize_results(
        pcd_path=args.pcd_path,
        poses_path=args.poses_path,
        image_folder=args.image_folder,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,
        port=args.port,
    )


if __name__ == "__main__":
    main()

