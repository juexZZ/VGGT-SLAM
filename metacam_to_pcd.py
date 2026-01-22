# convert metacam's .las file to pcd file and visualize it with viser for checking
import laspy
import numpy as np
import viser
import time
from pathlib import Path

def process_point_cloud(las_path, output_pcd_path):
    # 1. Read LAS file using laspy
    print(f"Reading {las_path}...")
    las = laspy.read(las_path)

    # Extract XYZ coordinates
    # las.xyz handles scaling and offsets automatically
    points = np.array(las.xyz)
    print(f"Points shape: {points.shape}")
    
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
    stride = 100
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
    # save_pcd(output_pcd_path, points, colors)
    # print(f"Saved PCD to {output_pcd_path}")

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

if __name__ == "__main__":
    # Replace with your actual file path
    input_las = "../metacam/8thfloor/8thfloor_small_static0/colorized.las" 
    output_pcd = "../metacam/8thfloor/8thfloor_small_static0/colorized.pcd"
    
    if Path(input_las).exists():
        process_point_cloud(input_las, output_pcd)
    else:
        print(f"File {input_las} not found.")