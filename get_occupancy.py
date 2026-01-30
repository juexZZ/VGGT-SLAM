"""
Occupancy map + camera trajectory visualization.

Inputs:
- A colorized point cloud `.pcd` (world frame). Used to build an occupancy map.
- A COLMAP `images.txt` file (world->cam extrinsics + image names).
- A `path.txt` file of image names (one per line, ordered).

Functionality:
1) Build an occupancy grid from the point cloud (exclude ceiling points above `--ceiling_z`, default 2.5m).
2) Read the COLMAP trajectory, extract poses for images listed in `path.txt`.
3) Visualize occupancy and trajectory in the same viser server:
   - Occupancy: cubes of size `--voxel_size` (default 0.05m).
     For each (x,y) cell we compute height range (max_z - min_z):
       - if > `--height_thresh` (default 0.10m), mark as NOT navigable (red)
       - else navigable (green)
   - Trajectory: start (green), end (red), intermediate (blue), and optional camera frames/frustums.
4) Determine navigability of each consecutive segment in XY:
   - sample points along the segment and check if any hit a non-navigable occupancy cell.
   - visualize segments in green (navigable) or red (blocked).
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import open3d as o3d
import viser
import viser.transforms as viser_tf


def get_T_zup_from_xleft_ydown_zin() -> np.ndarray:
    """
    Build a 4x4 transform that maps points/poses from the dataset coordinate system:
        x: left, y: down, z: inward(into camera)
    into a RIGHT-HANDED z-up coordinate system:
        X: right, Y: backward (into room), Z: up

    Mapping (point):
        X = -x
        Y = -z
        Z = -y
    """
    R = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def apply_T_world(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (T @ pts_h.T).T
    return (out[:, :3] / out[:, 3:]).astype(np.float32)


def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_colmap_images_txt_poses(images_txt_path: str) -> Dict[str, np.ndarray]:
    """
    Parse COLMAP `images.txt` and return cam2world 4x4 matrices keyed by image basename.

    COLMAP format (two lines per image):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[]...
    Pose is world->cam: X_c = R_cw X_w + t_cw
    cam center: C_w = -R_cw^T t_cw
    cam2world: R_wc = R_cw^T, t_wc = C_w
    """
    poses: Dict[str, np.ndarray] = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]
            except Exception:
                continue
            R_cw = _quat_wxyz_to_rotmat(qw, qx, qy, qz)
            t_cw = np.array([tx, ty, tz], dtype=np.float64)
            R_wc = R_cw.T
            C_w = -R_cw.T @ t_cw
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_wc
            T[:3, 3] = C_w
            basename = name.split("/")[-1]
            poses[basename] = T
    return poses


def load_path_list(path_txt: str) -> List[str]:
    names: List[str] = []
    with open(path_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(os.path.basename(s))
    return names


def build_occupancy_from_pointcloud(
    points_xyz: np.ndarray,
    voxel_size: float,
    ceiling_z: float,
    height_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D occupancy grid over (x,y) using point cloud heights.

    Returns:
      - centers_xyz: (M,3) cube centers (placed at z=min_z + voxel_size/2)
      - is_blocked: (M,) bool (True if height_range > height_thresh)
      - cell_keys:  (M,2) int64 (ix,iy) cell coords
      - minz:       (M,) float32 per-cell min z
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    # print(f"filter point cloud height into range [-0.2, {ceiling_z}]")
    pts = pts[pts[:, 2] <= ceiling_z]
    # pts = pts[pts[:, 2] >= -0.2]
    print(f"Points after height filtering: {pts.shape[0]}")
    if pts.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    ix = np.floor(pts[:, 0] / voxel_size).astype(np.int64)
    iy = np.floor(pts[:, 1] / voxel_size).astype(np.int64)
    keys = np.stack([ix, iy], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    m = uniq.shape[0]

    z = pts[:, 2].astype(np.float32)
    minz = np.full((m,), np.inf, dtype=np.float32)
    maxz = np.full((m,), -np.inf, dtype=np.float32)
    np.minimum.at(minz, inv, z)
    np.maximum.at(maxz, inv, z)

    height_range = maxz - minz
    print(f"Height range: {height_range.min()}, {height_range.max()}")
    is_blocked = height_range > float(height_thresh)
    
    centers = np.zeros((m, 3), dtype=np.float32)
    centers[:, 0] = (uniq[:, 0].astype(np.float32) + 0.5) * float(voxel_size)
    centers[:, 1] = (uniq[:, 1].astype(np.float32) + 0.5) * float(voxel_size)
    centers[:, 2] = minz + float(voxel_size) * 0.5
    return centers, is_blocked, uniq, minz


def segment_is_navigable(
    p0: np.ndarray,
    p1: np.ndarray,
    voxel_size: float,
    blocked_cells: Dict[Tuple[int, int], bool],
    unknown_is_free: bool = True,
) -> bool:
    """
    Check straight-line navigability in XY by sampling occupancy cells.
    """
    p0 = np.asarray(p0, dtype=np.float32).reshape(3)
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)
    d = float(np.linalg.norm(p1[:2] - p0[:2]))
    step = float(voxel_size) * 0.5
    n = max(2, int(np.ceil(d / step)) + 1)
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    for x, y in zip(xs, ys):
        key = (int(np.floor(x / voxel_size)), int(np.floor(y / voxel_size)))
        if key not in blocked_cells:
            if unknown_is_free:
                continue
            return False
        if blocked_cells[key]:
            return False
    return True

@dataclass
class NavigabilityResult:
    details: List[bool] # list of bool, True if the segment is navigable, False if blocked
    navigability: bool

def compute_navigability(
    pcd_path: str, 
    colmap_images_txt: str, 
    path_txt: str,
    voxel_size=0.2,
    ceiling_z=1.0,
    height_thresh=0.2,
    unknown_is_free=False,
) -> NavigabilityResult:
    f"""
    A function to call for evaluation job to compute navigability.
    No visualization is performed in this function. If you want the visualization on a trajectory, use the main function.
    Inputs:
    - pcd_path: path to the point cloud .pcd file, example: [metacam_data_folder]/colorized.pcd
    - colmap_images_txt: path to the COLMAP images.txt file, example: [metacam_data_folder]/sparse/0/images.txt
    - path_txt: path to the path.txt file (one image name per line, ordered)
    - voxel_size: occupancy cell size (meters)
    - ceiling_z: exclude points above this height (meters)
    - height_thresh: if max_z-min_z > this, mark cell blocked (meters)
    - unknown_is_free: treat unknown cells as free (default True).
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(pcd_path)
    if not os.path.exists(colmap_images_txt):
        raise FileNotFoundError(colmap_images_txt)
    if not os.path.exists(path_txt):
        raise FileNotFoundError(path_txt)

    print(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Coordinate transform (always): input is x-left, y-down, z-forward(inward),
    # we convert to z-up so occupancy logic can use Z as vertical.
    print("Transforming input data from (x left, y down, z forward/inward) to z-up (X right, Y forward, Z up).")
    T_zup = get_T_zup_from_xleft_ydown_zin()
    pts = apply_T_world(T_zup, pts)

    print(f"Building occupancy (voxel={voxel_size}m, ceiling_z={ceiling_z}m, height_thresh={height_thresh}m)")
    centers, blocked, cell_keys, minz = build_occupancy_from_pointcloud(
        pts,
        voxel_size=float(voxel_size),
        ceiling_z=float(ceiling_z),
        height_thresh=float(height_thresh),
    )
    print(f"Occupancy cells: {centers.shape[0]}  blocked: {int(blocked.sum())}  free: {int((~blocked).sum())}")

    blocked_cells: Dict[Tuple[int, int], bool] = {
        (int(k[0]), int(k[1])): bool(b) for k, b in zip(cell_keys, blocked)
    }
    cell_center_z: Dict[Tuple[int, int], float] = {
        (int(k[0]), int(k[1])): float(c[2]) for k, c in zip(cell_keys, centers)
    }

    # Load trajectory poses
    poses_by_name = parse_colmap_images_txt_poses(colmap_images_txt)
    path_names = load_path_list(path_txt)
    traj_T = []
    missing = 0
    for name in path_names:
        if name not in poses_by_name:
            missing += 1
            continue
        traj_T.append(poses_by_name[name])
    if missing > 0:
        print(f"[warn] Missing {missing}/{len(path_names)} images from COLMAP for path list.")
    if len(traj_T) < 2:
        raise RuntimeError("Need at least 2 poses from path.txt to visualize trajectory.")

    traj_T_np = np.stack(traj_T, axis=0)  # (N,4,4)
    # Apply the same world-frame transform to the camera poses.
    traj_T_np = (T_zup[None, :, :] @ traj_T_np).astype(np.float64)
    traj_pts = traj_T_np[:, :3, 3].astype(np.float32)
    # make sure the traj_pts not a blocked cell in occupancy grid
    # We only have a 2D occupancy grid keyed by (ix, iy). So map each traj point's (x,y)
    # into (ix,iy), then if that cell exists and is blocked, mark it free.
    cell_index: Dict[Tuple[int, int], int] = {
        (int(k[0]), int(k[1])): i for i, k in enumerate(cell_keys)
    }
    traj_ix = np.floor(traj_pts[:, 0] / float(voxel_size)).astype(np.int64)
    traj_iy = np.floor(traj_pts[:, 1] / float(voxel_size)).astype(np.int64)
    traj_xy_keys = {(int(ix), int(iy)) for ix, iy in zip(traj_ix.tolist(), traj_iy.tolist())}

    num_unblocked = 0
    num_missing = 0
    for key in traj_xy_keys:
        if key not in cell_index:
            num_missing += 1
            continue
        i = cell_index[key]
        if bool(blocked[i]):
            blocked[i] = False
            blocked_cells[key] = False
            num_unblocked += 1
    print(f"Unblocked {num_unblocked} occupancy cells under trajectory (missing cells: {num_missing}).")
    
    # Start viser
    print(f"Adding occupancy cubes... total number of cubes: {centers.shape[0]}")
    max_cubes = int(max_cubes)

    # Segment navigability + visualization
    print("Checking segment navigability...")
    # Place segment samples near the floor for visibility
    floor_z = float(np.percentile(pts[:, 2], 1)) if pts.shape[0] > 0 else 0.0
    seg_points_all = []
    seg_colors_all = []
    navigability_all = []

    for i in range(traj_pts.shape[0] - 1):
        p0 = traj_pts[i]
        p1 = traj_pts[i + 1]
        ok = segment_is_navigable(
            p0,
            p1,
            voxel_size=float(voxel_size),
            blocked_cells=blocked_cells,
            unknown_is_free=bool(unknown_is_free),
        )
        navigability_all.append(ok)

        # Visualize segment as sampled points snapped to occupancy cells.
        # For each (x,y) sample, look up the voxel cell:
        # - if free: color green, point at (x,y,z_cell+0.2)
        # - if blocked: color purple, point at (x,y,z_cell+0.2)
        dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
        step = float(voxel_size) * 0.5
        n = max(2, int(np.ceil(dxy / step)) + 1)
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
        xs = p0[0] + (p1[0] - p0[0]) * ts
        ys = p0[1] + (p1[1] - p0[1]) * ts
        seg_pts = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            key = (int(np.floor(x / float(voxel_size))), int(np.floor(y / float(voxel_size))))
            is_blk = blocked_cells.get(key, (not bool(unknown_is_free)))
            zc = cell_center_z.get(key, floor_z + float(voxel_size) * 0.5)
            seg_pts.append([x, y, float(zc) + 0.2])
        seg_pts = np.asarray(seg_pts, dtype=np.float32)
        seg_points_all.append(seg_pts)

        
    navigability_all = np.array(navigability_all)
    print("***** Trajectory Navigability *****")
    print(f"Total number of segments: {len(navigability_all)}")
    print(f"Number of navigable segments: {int(navigability_all.sum())}")
    print(f"Number of blocked segments: {int(navigability_all.shape[0] - navigability_all.sum())}")
    print(f"Navigability: {np.all(navigability_all)}")
    print("***********************************")
    
    return NavigabilityResult(
        details=navigability_all,
        navigability=np.all(navigability_all),
    )

    


def main():
    parser = argparse.ArgumentParser(description="Build occupancy map and visualize with trajectory in viser.")
    parser.add_argument("--pcd_path", type=str, required=True, help="Path to colorized point cloud .pcd")
    parser.add_argument("--colmap_images_txt", type=str, required=True, help="Path to COLMAP images.txt")
    parser.add_argument("--path_txt", type=str, required=True, help="Path to path.txt (one image name per line, ordered)")

    parser.add_argument("--voxel_size", type=float, default=0.2, help="Occupancy cell size (meters)")
    parser.add_argument("--ceiling_z", type=float, default=1.0, help="Exclude points above this height (meters)")
    parser.add_argument("--height_thresh", type=float, default=0.2, help="If max_z-min_z > this, mark cell blocked (meters)")
    parser.add_argument("--unknown_is_free", action="store_true", help="Treat unknown cells as free (default True).")

    parser.add_argument("--port", type=int, default=8090, help="Viser port")
    parser.add_argument("--max_cubes", type=int, default=60000, help="Cap occupancy cubes to render (for performance)")
    parser.add_argument("--cube_opacity", type=float, default=0.6, help="Opacity for occupancy cubes")
    parser.add_argument("--show_camera_frames", action="store_true", help="Add camera frames/frustums for the trajectory")
    parser.add_argument("--traj_point_size", type=float, default=0.1, help="Trajectory point size")
    parser.add_argument("--segment_point_size", type=float, default=0.01, help="Segment sample point size")
    args = parser.parse_args()

    if not os.path.exists(args.pcd_path):
        raise FileNotFoundError(args.pcd_path)
    if not os.path.exists(args.colmap_images_txt):
        raise FileNotFoundError(args.colmap_images_txt)
    if not os.path.exists(args.path_txt):
        raise FileNotFoundError(args.path_txt)

    print(f"Loading point cloud: {args.pcd_path}")
    pcd = o3d.io.read_point_cloud(args.pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Coordinate transform (always): input is x-left, y-down, z-forward(inward),
    # we convert to z-up so occupancy logic can use Z as vertical.
    print("Transforming input data from (x left, y down, z forward/inward) to z-up (X right, Y forward, Z up).")
    T_zup = get_T_zup_from_xleft_ydown_zin()
    pts = apply_T_world(T_zup, pts)

    print(f"Building occupancy (voxel={args.voxel_size}m, ceiling_z={args.ceiling_z}m, height_thresh={args.height_thresh}m)")
    centers, blocked, cell_keys, minz = build_occupancy_from_pointcloud(
        pts,
        voxel_size=float(args.voxel_size),
        ceiling_z=float(args.ceiling_z),
        height_thresh=float(args.height_thresh),
    )
    print(f"Occupancy cells: {centers.shape[0]}  blocked: {int(blocked.sum())}  free: {int((~blocked).sum())}")

    blocked_cells: Dict[Tuple[int, int], bool] = {
        (int(k[0]), int(k[1])): bool(b) for k, b in zip(cell_keys, blocked)
    }
    cell_center_z: Dict[Tuple[int, int], float] = {
        (int(k[0]), int(k[1])): float(c[2]) for k, c in zip(cell_keys, centers)
    }

    # Load trajectory poses
    poses_by_name = parse_colmap_images_txt_poses(args.colmap_images_txt)
    path_names = load_path_list(args.path_txt)
    traj_T = []
    missing = 0
    for name in path_names:
        if name not in poses_by_name:
            missing += 1
            continue
        traj_T.append(poses_by_name[name])
    if missing > 0:
        print(f"[warn] Missing {missing}/{len(path_names)} images from COLMAP for path list.")
    if len(traj_T) < 2:
        raise RuntimeError("Need at least 2 poses from path.txt to visualize trajectory.")

    traj_T_np = np.stack(traj_T, axis=0)  # (N,4,4)
    # Apply the same world-frame transform to the camera poses.
    traj_T_np = (T_zup[None, :, :] @ traj_T_np).astype(np.float64)
    traj_pts = traj_T_np[:, :3, 3].astype(np.float32)
    # breakpoint()
    # make sure the traj_pts not a blocked cell in occupancy grid
    # We only have a 2D occupancy grid keyed by (ix, iy). So map each traj point's (x,y)
    # into (ix,iy), then if that cell exists and is blocked, mark it free.
    cell_index: Dict[Tuple[int, int], int] = {
        (int(k[0]), int(k[1])): i for i, k in enumerate(cell_keys)
    }
    traj_ix = np.floor(traj_pts[:, 0] / float(args.voxel_size)).astype(np.int64)
    traj_iy = np.floor(traj_pts[:, 1] / float(args.voxel_size)).astype(np.int64)
    traj_xy_keys = {(int(ix), int(iy)) for ix, iy in zip(traj_ix.tolist(), traj_iy.tolist())}

    num_unblocked = 0
    num_missing = 0
    for key in traj_xy_keys:
        if key not in cell_index:
            num_missing += 1
            continue
        i = cell_index[key]
        if bool(blocked[i]):
            blocked[i] = False
            blocked_cells[key] = False
            num_unblocked += 1
    print(f"Unblocked {num_unblocked} occupancy cells under trajectory (missing cells: {num_missing}).")
    
    # Start viser
    print(f"Starting viser server on port {args.port} ...")
    server = viser.ViserServer(host="0.0.0.0", port=int(args.port))

    # Visualize occupancy cubes
    print(f"Adding occupancy cubes... total number of cubes: {centers.shape[0]}")
    max_cubes = int(args.max_cubes)
    if centers.shape[0] > max_cubes:
        idx = np.random.choice(centers.shape[0], size=max_cubes, replace=False)
        centers_vis = centers[idx]
        blocked_vis = blocked[idx]
        print(f"[warn] Subsampling occupancy cubes: {centers.shape[0]} -> {max_cubes}")
    else:
        centers_vis = centers
        blocked_vis = blocked

    # dims = (float(args.voxel_size), float(args.voxel_size), float(args.voxel_size))
    # color the blocked cells red, the free cells green
    colors_vis = np.zeros((centers_vis.shape[0], 3), dtype=np.float32)
    colors_vis[:] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    colors_vis[blocked_vis] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    server.scene.add_point_cloud(
        name="occupancy/cells",
        points=centers_vis,
        colors=colors_vis,
        point_size=float(args.voxel_size*0.8),
        point_shape="rounded",
    )
    # alternatively, visualize the but filter z value
    z_mask = pts[:, 2] <= args.ceiling_z
    vis_pts = pts[z_mask]
    vis_colors = np.asarray(pcd.colors)[z_mask]
    # Convert colors to uint8 if they're in [0, 1] range
    if vis_colors.max() <= 1.0:
        vis_colors = (vis_colors * 255).astype(np.uint8)
    else:
        vis_colors = vis_colors.astype(np.uint8)
    
    vis_stride = 4
    vis_pts = vis_pts[::vis_stride]
    vis_colors = vis_colors[::vis_stride]
    server.scene.add_point_cloud(
        name="occupancy/points",
        points=vis_pts,
        colors=vis_colors,
        point_size=float(args.voxel_size*0.5),
        point_shape="rounded",
    )

    # Visualize trajectory points
    print("Adding trajectory...")
    traj_colors = np.zeros((traj_pts.shape[0], 3), dtype=np.float32)
    if traj_pts.shape[0] >= 1:
        traj_colors[:] = np.array([1.0, 0.5, 0.0], dtype=np.float32)  # orange
        traj_colors[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # start blue
        traj_colors[-1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # end green
    server.scene.add_point_cloud(
        name="trajectory/points",
        points=traj_pts,
        colors=traj_colors,
        point_size=float(args.traj_point_size),
        point_shape="diamond",
    )

    # Optional camera frames/frustums
    if args.show_camera_frames:
        for i, T in enumerate(traj_T_np):
            T_world_cam = viser_tf.SE3.from_matrix(T[:3, :])
            server.scene.add_frame(
                name=f"trajectory/frame_{i}",
                wxyz=T_world_cam.rotation().wxyz,
                position=T_world_cam.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            # try add camera frustum (no image)
            
            server.scene.add_camera_frustum(
                name=f"trajectory/frustum_{i}",
                fov=1.0,
                aspect=1.0,
                scale=0.08,
                wxyz=T_world_cam.rotation().wxyz,
                position=T_world_cam.translation(),
                color=traj_colors[i],
            )

    # Segment navigability + visualization
    print("Checking segment navigability...")
    # Place segment samples near the floor for visibility
    floor_z = float(np.percentile(pts[:, 2], 1)) if pts.shape[0] > 0 else 0.0
    seg_points_all = []
    seg_colors_all = []
    navigability_all = []

    for i in range(traj_pts.shape[0] - 1):
        p0 = traj_pts[i]
        p1 = traj_pts[i + 1]
        ok = segment_is_navigable(
            p0,
            p1,
            voxel_size=float(args.voxel_size),
            blocked_cells=blocked_cells,
            unknown_is_free=bool(args.unknown_is_free),
        )
        navigability_all.append(ok)
        col = np.array([0.0, 1.0, 0.0], dtype=np.float32) if ok else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Visualize segment as sampled points snapped to occupancy cells.
        # For each (x,y) sample, look up the voxel cell:
        # - if free: color green, point at (x,y,z_cell+0.2)
        # - if blocked: color purple, point at (x,y,z_cell+0.2)
        dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
        step = float(args.voxel_size) * 0.5
        n = max(2, int(np.ceil(dxy / step)) + 1)
        ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
        xs = p0[0] + (p1[0] - p0[0]) * ts
        ys = p0[1] + (p1[1] - p0[1]) * ts
        seg_pts = []
        seg_cols = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            key = (int(np.floor(x / float(args.voxel_size))), int(np.floor(y / float(args.voxel_size))))
            is_blk = blocked_cells.get(key, (not bool(args.unknown_is_free)))
            zc = cell_center_z.get(key, floor_z + float(args.voxel_size) * 0.5)
            seg_pts.append([x, y, float(zc) + 0.2])
            if is_blk:
                seg_cols.append([0.6, 0.0, 0.8])  # purple
            else:
                seg_cols.append([0.0, 1.0, 0.0])  # green
        seg_pts = np.asarray(seg_pts, dtype=np.float32)
        seg_cols = np.asarray(seg_cols, dtype=np.float32)
        seg_points_all.append(seg_pts)
        seg_colors_all.append(seg_cols)

    if len(seg_points_all) > 0:
        seg_pts_all = np.concatenate(seg_points_all, axis=0)
        seg_cols_all = np.concatenate(seg_colors_all, axis=0)
        server.scene.add_point_cloud(
            name="trajectory/segments",
            points=seg_pts_all.astype(np.float32),
            colors=seg_cols_all.astype(np.float32),
            point_size=float(max(args.segment_point_size, float(args.voxel_size) * 0.4)),
            point_shape="circle",
        )
        
    navigability_all = np.array(navigability_all)
    print("***** Trajectory Navigability *****")
    print(f"Total number of segments: {len(navigability_all)}")
    print(f"Number of navigable segments: {int(navigability_all.sum())}")
    print(f"Number of blocked segments: {int(navigability_all.shape[0] - navigability_all.sum())}")
    print(f"Navigability: {np.all(navigability_all)}")
    print("***********************************")

    print("=" * 60)
    print(f"Visualization ready: http://localhost:{args.port}")
    print("Press Enter to exit...")
    try:
        input()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()