# convert metacam's .las file to pcd file and visualize it with viser for checking
import laspy
import numpy as np
import viser
import time
from pathlib import Path
import open3d as o3d

# Coordinate system transformation matrices
GLOBAL_ROT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0]
])

GLOBAL_TRANS = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

def apply_T_world(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got shape={pts.shape}")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (T @ pts_h.T).T
    out_xyz = out[:, :3] / np.clip(out[:, 3:4], 1e-12, None)
    return out_xyz.astype(np.float32)

def _extract_las_colors_uint8(las) -> np.ndarray | None:
    """
    Returns (N,3) uint8 colors if available, else None.
    LAS RGB is often uint16 in [0,65535]; we scale to [0,255].
    """
    has_rgb = all(hasattr(las, a) for a in ("red", "green", "blue"))
    if not has_rgb:
        return None
    r_raw = np.asarray(getattr(las, "red"))
    g_raw = np.asarray(getattr(las, "green"))
    b_raw = np.asarray(getattr(las, "blue"))
    max_val = float(max(r_raw.max(initial=0), g_raw.max(initial=0), b_raw.max(initial=0)))
    # If it's 16-bit, typical max is 65535; if already 8-bit, max ~255.
    scale = 255.0 / max_val if max_val > 255.0 and max_val > 0 else 1.0
    r = np.clip((r_raw * scale).astype(np.uint8), 0, 255)
    g = np.clip((g_raw * scale).astype(np.uint8), 0, 255)
    b = np.clip((b_raw * scale).astype(np.uint8), 0, 255)
    return np.stack([r, g, b], axis=-1)

def process_point_cloud(las_path, output_pcd_path):
    # 1. Read LAS file using laspy
    print(f"Reading {las_path}...")
    las = laspy.read(las_path)

    # Extract XYZ coordinates
    x = las.x
    y = las.y
    z = las.z
    
    points_homogeneous = np.column_stack([x, y, z, np.ones(len(x))])
    
    corrected_points = points_homogeneous.copy()
    corrected_points[:, :3] = corrected_points[:, :3] @ GLOBAL_ROT
    
    corrected_points = (GLOBAL_TRANS @ corrected_points.T).T
    
    y_rot_180 = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    corrected_points = (y_rot_180 @ corrected_points.T).T

    # Drop homogeneous coord for visualization / PCD writing.
    points = corrected_points[:, :3].astype(np.float32)
    
    # Extract colors if available (LAS colors are often 16-bit, need to scale to 8-bit)
    if hasattr(las, 'red'):
        red = (las.red / 65535.0 * 255).astype(np.uint8)
        green = (las.green / 65535.0 * 255).astype(np.uint8)
        blue = (las.blue / 65535.0 * 255).astype(np.uint8)
        colors = np.stack([red, green, blue], axis=-1)
    else:
        # Fallback to intensity-based grayscale if no RGB
        intensity = (las.intensity / np.max(las.intensity) * 255).astype(np.uint8)
        colors = np.stack([intensity] * 3, axis=-1)
    print(f"Colors shape: {colors.shape}")
    
    # visualize with a stride
    stride = 10
    subsampled_points = points[::stride]
    subsampled_colors = colors[::stride]
    print(f"Downsampled to {len(subsampled_points)} points (stride={stride})")
    
    # 2. Visualize with Viser
    server = viser.ViserServer()
    
    # # Center the cloud for better visualization in viser
    # offset = np.mean(points, axis=0)
    # centered_points = points - offset
    
    print(f"Visualizing at http://localhost:8080")
    server.scene.add_point_cloud(
        name="las_cloud",
        points=subsampled_points,
        colors=subsampled_colors,
        point_shape="circle",
        point_size=0.01,
    )

    # 3. Save as PCD (Simple Header Format)
    save_pcd(output_pcd_path, points, colors)
    print(f"Saved PCD to {output_pcd_path}")

    # Keep server alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped.")

def save_pcd(path, points, colors=None):
    """Saves a simple ASCII PCD file."""
    num_points = points.shape[0]
    with open(path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write("VERSION 0.7\n")
        if colors is not None:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
        else:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
        for i in range(num_points):
            x, y, z = points[i]
            if colors is not None:
                # Pack RGB into a single float/int for PCD format
                r, g, b = colors[i]
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                f.write(f"{x} {y} {z} {rgb}\n")
            else:
                f.write(f"{x} {y} {z}\n")


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

def convert_las_to_ply(
    scene_path: Path,
    las_name: str = "colorized.las",
    ply_name: str = "colorized_z_up.ply",
    transform: str = "zup",
) -> Path | None:
    """
    Convert a LAS point cloud to a PLY (Open3D) point cloud.

    Transforms:
      - transform="legacy": apply the GLOBAL_ROT/GLOBAL_TRANS/y-180 pipeline. In your setup this
        yields a camera-convention coordinate frame: (x left, y down, z into camera).
      - transform="zup": first apply the "legacy" pipeline (to get camera-convention),
        then convert camera-convention -> right-handed z-up via `get_T_zup_from_xleft_ydown_zin()`.
        This matches the convention expected by SpatialLM.
      - transform="identity": write as-is.
    """
    scene_path = Path(scene_path)
    las_path = scene_path / las_name
    if not las_path.exists():
        print(f"Warning: LAS file not found at {las_path}")
        return None

    print(f"\n=== Converting LAS -> PLY ({transform}) for {scene_path.name} ===")
    las = laspy.read(str(las_path))

    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64)  # (N,3)

    pts_h = np.column_stack([pts, np.ones((pts.shape[0],), dtype=np.float64)])
    pts_h[:, :3] = pts_h[:, :3] @ GLOBAL_ROT
    pts_h = (GLOBAL_TRANS @ pts_h.T).T
    y_rot_180 = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_h = (y_rot_180 @ pts_h.T).T
    pts = pts_h[:, :3].astype(np.float32)  # camera convention: x-left, y-down, z-in
    if transform == "zup":
        T = get_T_zup_from_xleft_ydown_zin()
        pts = apply_T_world(T, pts)

    colors_u8 = _extract_las_colors_uint8(las)
    ply_path = scene_path / ply_name
    if colors_u8 is None:
        write_ply_file(ply_path, pts)
    else:
        write_ply_file(ply_path, pts, colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2])
    print(f"Created PLY file: {ply_path}")
    return ply_path
    
def write_ply_file(ply_path, points, r=None, g=None, b=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if r is not None and g is not None and b is not None:
        colors = np.column_stack([r, g, b]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        num_points = len(points)
        white_colors = np.ones((num_points, 3))
        pcd.colors = o3d.utility.Vector3dVector(white_colors)
    
    o3d.io.write_point_cloud(str(ply_path), pcd)

if __name__ == "__main__":
    # Replace with your actual file path
    input_las = "../metacam/8thfloor/8thfloor_small_static0/colorized.las" 
    output_pcd = "../metacam/8thfloor/8thfloor_small_static0/colorized.pcd"
    output_ply = "../metacam/8thfloor/8thfloor_small_static0/colorized_z_up.ply"
    
    convert_las_to_ply(
        scene_path=Path(input_las).parent,
        las_name=Path(input_las).name,
        ply_name=Path(output_ply).name,
        transform="zup",
    )
    
    # if Path(input_las).exists():
    #     process_point_cloud(input_las, output_pcd)
    # else:
    #     print(f"File {input_las} not found.")