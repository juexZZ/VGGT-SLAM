#!/usr/bin/env python3
"""
Undistort MetaCam fisheye images (left/right) using fixed intrinsics/distortion.

This script is intentionally standalone: it only needs an input image folder and will
write undistorted images to an output folder. Extrinsics (transform matrices) are ignored.

Supported input layouts:
  1) <input_dir>/left/*.jpg|png and <input_dir>/right/*.jpg|png
  2) <input_dir>/* with filenames starting with "left_" or "right_"

By default, it outputs square pinhole images at 1600x1600 with 90° horizontal FOV.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class FisheyeModel:
    # Reference resolution that intrinsics were calibrated at
    w_ref: int
    h_ref: int
    fx_ref: float
    fy_ref: float
    cx_ref: float
    cy_ref: float
    # OpenCV fisheye distortion (k1,k2,k3,k4)
    k1: float
    k2: float
    k3: float
    k4: float

    def scaled_camera_matrix(self, actual_w: int, actual_h: int) -> np.ndarray:
        """Scale reference intrinsics to match actual image resolution."""
        sx = float(actual_w) / float(self.w_ref)
        sy = float(actual_h) / float(self.h_ref)
        return np.array(
            [
                [self.fx_ref * sx, 0.0, self.cx_ref * sx],
                [0.0, self.fy_ref * sy, self.cy_ref * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def distortion(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)


# Params copied from user-provided transforms.json snippets
LEFT_MODEL = FisheyeModel(
    w_ref=3040,
    h_ref=4032,
    fx_ref=1187.095159186288,
    fy_ref=1187.3641658709835,
    cx_ref=1582.466806267845,
    cy_ref=2037.5621301664378,
    k1=-0.010206811064634946,
    k2=-0.002676612556500302,
    k3=0.00020819087272026367,
    k4=-0.0004558519912419938,
)

RIGHT_MODEL = FisheyeModel(
    w_ref=3040,
    h_ref=4032,
    fx_ref=1186.9087929758348,
    fy_ref=1186.2272064372953,
    cx_ref=1597.404519695444,
    cy_ref=1994.1936442001027,
    k1=-0.008061384087927215,
    k2=-0.005394217768337191,
    k3=0.0022551527769710004,
    k4=-0.0009518699170852251,
)


def _iter_images(dir_path: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def discover_inputs(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Return {"left": [...], "right": [...]}.
    Supports either left/right subfolders or left_/right_ filename prefixes.
    """
    # Common layouts:
    # - <input>/left and <input>/right
    # - <input>/camera/left and <input>/camera/right  (MetaCam raw dumps)
    candidates = [
        (input_dir / "left", input_dir / "right"),
        (input_dir / "camera" / "left", input_dir / "camera" / "right"),
    ]
    for left_dir, right_dir in candidates:
        if left_dir.is_dir() or right_dir.is_dir():
            left_imgs = _iter_images(left_dir) if left_dir.is_dir() else []
            right_imgs = _iter_images(right_dir) if right_dir.is_dir() else []
            if left_imgs or right_imgs:
                return {"left": left_imgs, "right": right_imgs}

    all_imgs = _iter_images(input_dir)
    left_imgs = [p for p in all_imgs if p.name.lower().startswith("left_")]
    right_imgs = [p for p in all_imgs if p.name.lower().startswith("right_")]
    return {"left": left_imgs, "right": right_imgs}


def make_target_K(out_size: int, fov_deg: float) -> np.ndarray:
    """
    Square pinhole intrinsics for a desired output FOV (horizontal).
    For square, fx=fy and cx=cy=out_size/2.
    """
    if out_size <= 0:
        raise ValueError("out_size must be > 0")
    if not (0.0 < fov_deg < 180.0):
        raise ValueError("fov_deg must be in (0, 180)")
    fov_rad = np.deg2rad(fov_deg)
    f = (0.5 * out_size) / np.tan(0.5 * fov_rad)
    c = 0.5 * out_size
    return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_map_for_camera(
    model: FisheyeModel, sample_image_path: Path, target_K: np.ndarray, out_size: int
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    img = cv2.imread(str(sample_image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read sample image: {sample_image_path}")
    actual_h, actual_w = img.shape[:2]
    K = model.scaled_camera_matrix(actual_w=actual_w, actual_h=actual_h)
    D = model.distortion()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), target_K, (out_size, out_size), cv2.CV_16SC2
    )
    return map1, map2, (actual_w, actual_h)


def undistort_images(
    image_paths: Iterable[Path],
    map1: np.ndarray,
    map2: np.ndarray,
    output_dir: Path,
    overwrite: bool,
    keep_ext: bool,
    interpolation: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src_path in tqdm(list(image_paths), desc=f"Undistorting -> {output_dir.name}"):
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        out_name = src_path.name if keep_ext else (src_path.stem + ".png")
        dst_path = output_dir / out_name
        if dst_path.exists() and not overwrite:
            continue

        undistorted = cv2.remap(img, map1, map2, interpolation=interpolation)
        ok = cv2.imwrite(str(dst_path), undistorted)
        if ok:
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Undistort MetaCam fisheye images from a folder (left/right)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing images. Either has left/ and right/ subfolders, or files prefixed left_ / right_.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder. Default: <input_dir>/undistorted",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=1600,
        help="Output square image size (pixels). Default: 1600",
    )
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=90.0,
        help="Horizontal FOV (degrees) for output pinhole camera. Default: 90",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--keep_ext",
        action="store_true",
        help="Keep original file extension instead of writing PNG.",
    )
    parser.add_argument(
        "--interp",
        choices=["nearest", "linear", "cubic", "lanczos"],
        default="linear",
        help="Interpolation for remap. Default: linear",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    output_dir: Path = args.output_dir if args.output_dir is not None else (input_dir / "undistorted")

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interp_map[args.interp]

    groups = discover_inputs(input_dir)
    if len(groups["left"]) == 0 and len(groups["right"]) == 0:
        raise SystemExit(
            f"No images found in {input_dir}. Expected left/right subfolders or left_*/right_* files."
        )
    print(f"Discovered {len(groups['left'])} left images and {len(groups['right'])} right images in: {input_dir}")

    target_K = make_target_K(out_size=args.out_size, fov_deg=args.fov_deg)

    total_saved = 0
    if groups["left"]:
        map1, map2, _ = compute_map_for_camera(
            LEFT_MODEL, groups["left"][0], target_K=target_K, out_size=args.out_size
        )
        total_saved += undistort_images(
            image_paths=groups["left"],
            map1=map1,
            map2=map2,
            output_dir=(
                output_dir / "left"
                if (input_dir / "left").is_dir() or (input_dir / "camera" / "left").is_dir()
                else output_dir
            ),
            overwrite=args.overwrite,
            keep_ext=args.keep_ext,
            interpolation=interpolation,
        )

    if groups["right"]:
        map1, map2, _ = compute_map_for_camera(
            RIGHT_MODEL, groups["right"][0], target_K=target_K, out_size=args.out_size
        )
        total_saved += undistort_images(
            image_paths=groups["right"],
            map1=map1,
            map2=map2,
            output_dir=(
                output_dir / "right"
                if (input_dir / "right").is_dir() or (input_dir / "camera" / "right").is_dir()
                else output_dir
            ),
            overwrite=args.overwrite,
            keep_ext=args.keep_ext,
            interpolation=interpolation,
        )

    print(f"Done. Wrote {total_saved} undistorted images to: {output_dir}")


if __name__ == "__main__":
    # Avoid OpenCV multithreading oversubscription when used on big folders.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

