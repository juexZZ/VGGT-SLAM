import re
import os
import cv2
import torch
import numpy as np
import open3d as o3d

from vggt_slam.semantic_voxel import SemanticVoxel

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.H_world_map = None
        self.R_world_map = None
        self.poses = None
        self.frames = None
        self.vggt_intrinscs = None
        self.retrieval_vectors = None
        self.colors = None # (S, H, W, 3)
        self.conf = None # (S, H, W)
        self.conf_masks = None # (S, H, W)
        self.conf_threshold = None
        self.pointclouds = None # (S, H, W, 3)
        self.voxelized_points = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
        self.frame_names = None  # list[str] (basename per frame_id)
        self.frame_id_to_name = None  # dict[str, str] mapping str(frame_id) -> basename
        self.semantic_embeddings = None  # (S, H, W, d)
    
    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics):
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(self.conf, conf_threshold_percentile)
        self.vggt_intrinscs = intrinsics

    def add_all_semantic_embeddings(self, semantic_embeddings: np.ndarray):
        """
        Store dense semantic embeddings aligned to the per-frame point maps.

        Args:
            semantic_embeddings: (S, H, W, d) array.
        """
        if semantic_embeddings is None:
            self.semantic_embeddings = None
            return

        if not isinstance(semantic_embeddings, np.ndarray):
            raise TypeError("semantic_embeddings must be a numpy array of shape (S,H,W,d)")

        if semantic_embeddings.ndim != 4:
            raise ValueError(f"semantic_embeddings must have 4 dims (S,H,W,d), got shape={semantic_embeddings.shape}")

        if self.pointclouds is not None:
            if semantic_embeddings.shape[0] != self.pointclouds.shape[0] or semantic_embeddings.shape[1] != self.pointclouds.shape[1] or semantic_embeddings.shape[2] != self.pointclouds.shape[2]:
                raise ValueError(
                    "semantic_embeddings spatial dims must match pointclouds. "
                    f"semantic={semantic_embeddings.shape[:3]} vs points={self.pointclouds.shape[:3]}"
                )
        self.semantic_embeddings = semantic_embeddings
        print(f"Added semantic embeddings shape: {semantic_embeddings.shape} to submap {self.submap_id}")
            
    def add_all_frames(self, frames):
        self.frames = frames
    
    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold
    
    def get_frame_at_index(self, index):
        return self.frames[index, ...]
    
    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_all_frames(self):
        return self.frames
    
    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_all_poses_world(self, ignore_loop_closure_frames=False):
        projection_mat_list = self.vggt_intrinscs @ np.linalg.inv(self.poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
        poses = []
        for index, projection_mat in enumerate(projection_mat_list):
            cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
            # print("cal", cal/cal[2,2])
            trans = trans/trans[3,0] # TODO see if we should normalize the rotation too with this.
            pose = np.eye(4)
            pose[0:3, 0:3] = np.linalg.inv(rot)
            pose[0:3,3] = trans[0:3,0]
            poses.append(pose)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break
        return np.stack(poses, axis=0)
    
    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]

    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names, 
        removing any leading zeros, and add them all to a list.

        Note: This does not include any of the loop closure frames.
        """
        frame_ids = []
        frame_names = []
        frame_id_to_name = {}
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                fid = float(match.group())
                frame_ids.append(fid)
                frame_names.append(filename)
                frame_id_to_name[str(fid)] = filename
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids
        self.frame_names = frame_names
        self.frame_id_to_name = frame_id_to_name

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map
    
    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def get_reference_homography(self):
        return self.H_world_map

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])
    
    def get_frame_ids(self):
        # Note this does not include any of the loop closure frames
        return self.frame_ids

    def filter_data_by_confidence(self, data, stride = 1):
        if stride == 1:
            init_conf_mask = self.conf >= self.conf_threshold
            return data[init_conf_mask]
        else:
            conf_sub = self.conf[:, ::stride, ::stride]
            data_sub = data[:, ::stride, ::stride, :]

            init_conf_mask = conf_sub >= self.conf_threshold
            return data_sub[init_conf_mask]

    def get_points_list_in_world_frame(self, ignore_loop_closure_frames=False):
        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        for index,points in enumerate(self.pointclouds):
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (self.H_world_map @ points_homogeneous.T).T
            point_list.append((points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape))
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] >= self.conf_threshold
            frame_conf_mask.append(conf_mask)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break
        return point_list, frame_id_list, frame_conf_mask

    def get_points_in_world_frame(self, stride = 1):
        points = self.filter_data_by_confidence(self.pointclouds, stride)

        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        return points_transformed[:, :3] / points_transformed[:, 3:]

    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                points = self.filter_data_by_confidence(self.pointclouds)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_confidence(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame
    
    def get_points_colors(self, stride = 1):
        colors = self.filter_data_by_confidence(self.colors, stride)
        return colors.reshape(-1, 3)

    def get_semantic_voxel_in_world_frame(
        self,
        voxel_size: float,
        stride: int = 1,
        ignore_loop_closure_frames: bool = False,
    ) -> SemanticVoxel:
        """
        Project dense per-pixel semantic embeddings onto 3D points, transform to world frame,
        then voxel-average features.

        - Points are filtered by the confidence threshold (same policy as pointcloud export).
        - Voxel keys are computed as floor(point_world / voxel_size).
        - Features of all points in the same voxel are averaged.
        - Provenance is tracked as contributing (submap_id, frame_idx) pairs per voxel.
        """
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be > 0")
        if self.pointclouds is None:
            raise RuntimeError("No pointclouds in submap. Run add_all_points() first.")
        if self.semantic_embeddings is None:
            raise RuntimeError("No semantic embeddings in submap. Run add_all_semantic_embeddings() first.")
        if self.H_world_map is None:
            raise RuntimeError("No reference homography in submap. Run set_reference_homography() first.")

        # Optionally truncate frames at the last non-loop frame.
        end_idx = self.pointclouds.shape[0]
        if ignore_loop_closure_frames and (self.last_non_loop_frame_index is not None):
            end_idx = min(end_idx, self.last_non_loop_frame_index + 1)

        pts = self.pointclouds[:end_idx]
        sem = self.semantic_embeddings[:end_idx]
        conf = self.conf[:end_idx]

        # Confidence mask
        mask = conf >= self.conf_threshold  # (S, H, W)

        # Flatten points/features under mask
        pts_flat = pts[mask]  # (N, 3)
        sem_flat = sem[mask]  # (N, d)
        if pts_flat.shape[0] == 0:
            return SemanticVoxel(
                voxel_size=voxel_size,
                centers_world=np.zeros((0, 3), dtype=np.float32),
                features=np.zeros((0, sem.shape[-1]), dtype=np.float32),
                contributors=[],
            )

        # Build per-point frame_idx provenance aligned to mask flattening order.
        # (S,H,W) -> flat by boolean mask (row-major).
        frame_idx_grid = np.broadcast_to(
            np.arange(end_idx, dtype=np.int32)[:, None, None],
            mask.shape,
        )
        frame_idx_flat = frame_idx_grid[mask].astype(np.int32)

        # Transform points into world frame using H_world_map (same as get_points_in_world_frame()).
        pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)], axis=1)  # (N, 4)
        pts_w_h = (self.H_world_map @ pts_h.T).T  # (N, 4)
        pts_world = (pts_w_h[:, :3] / pts_w_h[:, 3:]).astype(np.float32)  # (N, 3)

        # Voxelize: compute integer voxel coordinates.
        voxel_coords = np.floor(pts_world / voxel_size).astype(np.int64)  # (N, 3)
        unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        num_vox = unique_coords.shape[0]

        # Accumulate feature sums and counts.
        d = sem_flat.shape[-1]
        feat_sum = np.zeros((num_vox, d), dtype=np.float32)
        counts = np.zeros((num_vox,), dtype=np.int64)
        np.add.at(feat_sum, inverse, sem_flat.astype(np.float32))
        np.add.at(counts, inverse, 1)
        feat_avg = feat_sum / counts[:, None]        # Voxel centers (use cell center in world frame).
        centers_world = ((unique_coords.astype(np.float32) + 0.5) * voxel_size).astype(np.float32)

        # Provenance: list contributors per voxel.
        contributors = [[] for _ in range(num_vox)]
        submap_id = int(self.submap_id)
        for p_i, v_i in enumerate(inverse.tolist()):
            # frame_ids are floats extracted from filenames; store as string for easier external lookup.
            if self.frame_ids is not None and int(frame_idx_flat[p_i]) < len(self.frame_ids):
                frame_id = str(self.frame_ids[int(frame_idx_flat[p_i])])
            else:
                frame_id = str(int(frame_idx_flat[p_i]))
            contributors[v_i].append((submap_id, frame_id))

        return SemanticVoxel(
            voxel_size=voxel_size,
            centers_world=centers_world,
            features=feat_avg,
            contributors=contributors,
        )
