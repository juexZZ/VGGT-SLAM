from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return 3x3 rotation matrix from quaternion (w,x,y,z)."""
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


def parse_colmap_images_txt(images_txt_path: str) -> Dict[str, np.ndarray]:
    """
    Parse COLMAP `images.txt` and return camera centers keyed by image basename.

    COLMAP format (two lines per image):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[]...

    The pose is world->cam: X_cam = R_cw * X_world + t_cw
    Camera center in world: C = -R_cw^T * t_cw
    """
    centers: Dict[str, np.ndarray] = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                # likely a POINTS2D line; skip
                continue
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]
            except Exception:
                continue
            R_cw = _quat_wxyz_to_rotmat(qw, qx, qy, qz)
            t_cw = np.array([tx, ty, tz], dtype=np.float64)
            C_w = -R_cw.T @ t_cw
            basename = name.split("/")[-1]
            centers[basename] = C_w.astype(np.float64)
    return centers


@dataclass
class Sim3:
    s: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.s * self.R
        T[:3, 3] = self.t
        return T


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """
    Estimate Sim(3) aligning src->dst using Umeyama (least squares).

    Finds s,R,t such that: dst ~= s * R * src + t
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src and dst must be Nx3 and same shape; got {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 point correspondences for Sim(3) alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)

    R = U @ Vt
    # Fix improper rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    if with_scale:
        var_src = (X * X).sum() / n
        s = float(np.sum(S) / (var_src + 1e-12))
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return Sim3(s=s, R=R, t=t)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

