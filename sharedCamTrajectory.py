"""
sharedCamTrajectory.py — Shared Plücker Trajectory Interface for CamLoop4D
============================================================================
Implements the shared-camera-trajectory interface described in the CamLoop4D
paper (Pacific Graphics 2026, paper ID 1170, Section 3.2.3):

    "both stages consume the same Plücker trajectory tensor, removing pose
     re-estimation by construction."

This script is the *one-way convention* between the two stages:

  [1] Video Generation  (vgen/)  -- produces the specified camera trajectory
       as a Plücker trajectory tensor, or reads a user-specified trajectory.
  [2] Shared file        -- writes the trajectory to a camera file that the
       reconstructor reads directly (no pose re-estimation).
  [3] 4D Reconstruction (flow3d) -- `CasualDataset.load_cameras` consumes the
       same file via `--data.camera-type droid_recon`.

Output file format (identical to `flow3d/data/casual_dataset.py::load_cameras`):

    {
        "traj_c2w"   : np.ndarray (N, 4, 4),  # camera-to-world (world frame)
        "img_shape"  : tuple (h, w),
        "intrinsics" : np.ndarray (4,),       # (fx, fy, cx, cy)
        "tstamps"    : np.ndarray (N,),       # keyframe timestamps
    }

Usage:
    # 1) Procedural trajectory from a reference camera (Plücker from K|R|t)
    python sharedCamTrajectory.py plucker-trajectory \
        --ref-w2c ref_w2c.npy --fx 700 --fy 700 --cx 320 --cy 240 \
        --traj-type spiral --num-frames 80 \
        --out-dir ./data --seq-name my_scene

    # 2) Convert an existing trajectory file (e.g. exported from a generator)
    #    into the reconstructor camera file
    python sharedCamTrajectory.py convert \
        --traj-c2w traj_c2w.npy --intrinsics "700,700,320,240" --img-shape "480,720" \
        --out-dir ./data --seq-name my_scene

    # 3) Convert a RealEstate10K pose file into the iPhone-style camera JSONs
    #    (<out_dir>/camera/<frame_i>.json) -- the format consumed by
    #    iPhoneDataset (camera_type="original").
    python sharedCamTrajectory.py realestate10k-to-iphone \
        --pose-file pose_files/0000cc6d8b108390.txt \
        --out-dir ./data --seq-name my_scene

    # 4) Inspect an existing camera file
    python sharedCamTrajectory.py inspect --cam-file ./data/droid_recon/my_scene.npy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Plücker embeddings (paper Section 3.1, Eq. 1-2)
# ---------------------------------------------------------------------------


def compute_plucker(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int
) -> np.ndarray:
    """
    Build a Plücker embedding for a single camera.

    p_{u,v} = [ o x d_{u,v} ; d_{u,v} ] in R^6

    where o is the camera center in world coordinates and d_{u,v} is the
    normalized ray direction computed from intrinsics K and extrinsics [R|t]:

        d_{u,v} = normalize( R K^{-1} [u, v, 1]^T )

    Args:
        K: (3, 3) intrinsics.
        R: (3, 3) world-to-camera rotation (or camera-to-world; see note).
        t: (3,)   world-to-camera translation.
        H: image height.
        W: image width.

    Returns:
        np.ndarray (H, W, 6): per-pixel Plücker embedding.
    """
    inv_K = np.linalg.inv(K)
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    ones = np.ones_like(uu)
    # (H, W, 3) homogeneous pixel coordinates
    pix = np.stack([uu, vv, ones], axis=-1)
    # ray directions in camera frame
    dirs_cam = pix @ inv_K.T  # (H, W, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    # camera center in world coordinates: o = -R^T t (R,t camera-to-world) or
    # o = -inv(R) @ t. We keep the caller's convention consistent: R,t should
    # map world -> camera, so o = -R^T t.
    center = -R.T @ t
    dirs_world = dirs_cam @ R.T  # rotate to world frame
    moment = np.cross(np.broadcast_to(center, dirs_world.shape), dirs_world)
    return np.concatenate([moment, dirs_world], axis=-1)  # (H, W, 6)


def plucker_for_trajectory(
    traj_w2c: np.ndarray, Ks: np.ndarray, H: int, W: int
) -> np.ndarray:
    """
    Plücker embeddings for a whole trajectory.

    Args:
        traj_w2c: (N, 4, 4) world-to-camera poses.
        Ks: (N, 3, 3) intrinsics (or a single (3, 3) broadcast to all frames).

    Returns:
        np.ndarray (N, H, W, 6)
    """
    N = traj_w2c.shape[0]
    if Ks.ndim == 2:
        Ks = np.tile(Ks[None], (N, 1, 1))
    embs = []
    for i in range(N):
        R = traj_w2c[i, :3, :3]
        t = traj_w2c[i, :3, 3]
        embs.append(compute_plucker(Ks[i], R, t, H, W))
    return np.stack(embs, axis=0)


# ---------------------------------------------------------------------------
# Trajectory generators (re-implemented lightweight, mirroring
# flow3d/trajectories.py so this script is standalone)
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def get_lookat_w2cs(
    positions: np.ndarray, lookat: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Build world-to-camera matrices looking at `lookat` from `positions`."""
    positions = np.asarray(positions, dtype=np.float64)
    N = len(positions)
    w2cs = np.zeros((N, 4, 4))
    w2cs[:, 3, 3] = 1.0
    for i in range(N):
        z_axis = _normalize(lookat - positions[i])  # camera looks along +z? -> -z in w2c
        x_axis = _normalize(np.cross(up, z_axis))
        y_axis = np.cross(z_axis, x_axis)
        # w2c rotation (rows are camera axes in world coords)
        w2cs[i, :3, 0] = x_axis
        w2cs[i, :3, 1] = y_axis
        w2cs[i, :3, 2] = z_axis
        w2cs[i, :3, 3] = -np.array(
            [x_axis @ positions[i], y_axis @ positions[i], z_axis @ positions[i]]
        )
    return w2cs


def generate_trajectory(
    traj_type: str,
    num_frames: int,
    ref_w2c: np.ndarray | None = None,
    radius: float = 1.0,
    lookat: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate a camera trajectory around a reference camera.

    Supported types: spiral | orbit | lemniscate | arc | wander | fixed
    """
    if ref_w2c is not None:
        ref_w2c = np.asarray(ref_w2c, dtype=np.float64)
        center = ref_w2c[:3, 3] + radius * ref_w2c[:3, 2]  # look-at point in front
        if lookat is None:
            lookat = center
        if up is None:
            up = ref_w2c[:3, 1]
    if lookat is None:
        lookat = np.zeros(3)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    t = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)

    if traj_type == "spiral":
        rads = radius * np.ones(num_frames)
        z = np.linspace(-radius * 0.5, radius * 0.5, num_frames)
        pos = np.stack(
            [rads * np.cos(t), rads * np.sin(t), z], axis=-1
        )
        base = np.array([0.0, 0.0, 0.0])
    elif traj_type == "orbit":
        rads = radius * np.ones(num_frames)
        pos = np.stack(
            [rads * np.cos(t), rads * np.sin(t), np.zeros(num_frames)], axis=-1
        )
        base = np.array([0.0, 0.0, 0.0])
    elif traj_type == "lemniscate":
        scale = radius
        x = scale * np.cos(t) / (1.0 + np.sin(t) ** 2)
        y = scale * np.sin(t) * np.cos(t) / (1.0 + np.sin(t) ** 2)
        pos = np.stack([x, y, np.zeros(num_frames)], axis=-1)
        base = np.array([0.0, 0.0, 0.0])
    elif traj_type == "arc":
        angle = np.linspace(-radius, radius, num_frames)  # radius acts as degree
        pos = np.stack(
            [np.sin(np.deg2rad(angle)), np.zeros(num_frames), np.cos(np.deg2rad(angle))],
            axis=-1,
        )
        pos = pos * (radius * 2.0)
        base = np.array([0.0, 0.0, 0.0])
    elif traj_type == "wander":
        pos = np.stack(
            [
                radius * 0.5 * np.sin(t * 0.5),
                radius * 0.3 * np.cos(t * 0.5),
                np.zeros(num_frames),
            ],
            axis=-1,
        )
        base = np.array([0.0, 0.0, 0.0])
    elif traj_type == "fixed":
        pos = np.zeros((num_frames, 3))
        base = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    positions = base[None] + pos
    # put the trajectory around the reference camera's position
    if ref_w2c is not None:
        R_ref = ref_w2c[:3, :3]
        t_ref = ref_w2c[:3, 3]
        positions = positions @ R_ref.T + t_ref
    return get_lookat_w2cs(positions, lookat, up)


# ---------------------------------------------------------------------------
# Camera-file I/O (reconstructor-compatible format)
# ---------------------------------------------------------------------------


def save_camera_file(
    traj_c2w: np.ndarray,
    intrinsics: Sequence[float],
    img_shape: tuple[int, int],
    tstamps: np.ndarray | None,
    out_dir: str,
    seq_name: str,
) -> Path:
    """
    Write the shared camera file consumed by the reconstructor:
        <out_dir>/droid_recon/<seq_name>.npy
    """
    out_path = Path(out_dir) / "droid_recon" / f"{seq_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tstamps is None:
        tstamps = np.arange(len(traj_c2w))

    recon = {
        "traj_c2w": np.asarray(traj_c2w, dtype=np.float32),  # (N, 4, 4)
        "img_shape": (int(img_shape[0]), int(img_shape[1])),  # (h, w)
        "intrinsics": np.asarray(intrinsics, dtype=np.float32),  # (fx, fy, cx, cy)
        "tstamps": np.asarray(tstamps),
    }
    np.save(out_path, recon)
    print(f"[sharedCamTrajectory] saved shared camera file: {out_path}")
    print(f"    traj_c2w  : {recon['traj_c2w'].shape}")
    print(f"    img_shape : {recon['img_shape']}")
    print(f"    intrinsics: {recon['intrinsics']}")
    return out_path


def inspect_camera_file(cam_file: str) -> None:
    """Print the contents of a shared camera file (droid_recon format)."""
    cam_file = Path(cam_file)
    assert cam_file.exists(), f"Camera file {cam_file} does not exist."
    recon = np.load(cam_file, allow_pickle=True).item()
    print(f"[sharedCamTrajectory] inspect {cam_file}")
    for k, v in recon.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# RealEstate10K -> iPhone camera format
# ---------------------------------------------------------------------------


def read_realestate10k_poses(pose_file: str) -> list[dict]:
    """
    Read a RealEstate10K pose file (one camera per line).

    Line format (whitespace separated):
        <frame_id> <fx> <fy> <cx> <cy> <H> <W>
        <w2c[0][0]> <w2c[0][1]> <w2c[0][2]> <w2c[0][3]>
        <w2c[1][0]> <w2c[1][1]> <w2c[1][2]> <w2c[1][3]>
        <w2c[2][0]> <w2c[2][1]> <w2c[2][2]> <w2c[2][3]>
    (first line is a header and is skipped; 12 values follow the first 7 fields)

    Args:
        pose_file: path to the RealEstate10K .txt pose file.

    Returns:
        List of dicts, one per frame:
            {"fx", "fy", "cx", "cy", "H", "W", "w2c": (4, 4)}
    """
    with open(pose_file, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    cams = []
    for ln in lines[1:]:  # skip header
        vals = [float(x) for x in ln.strip().split(" ")]
        assert len(vals) == 7 + 12, f"Unexpected pose line length: {len(vals)}"
        fx, fy, cx, cy = vals[1:5]
        H, W = int(vals[5]), int(vals[6])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = np.array(vals[7:]).reshape(3, 4)
        cams.append(
            {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "H": H, "W": W, "w2c": w2c}
        )
    return cams


def realestate10k_to_iphone(
    pose_file: str, out_dir: str, seq_name: str
) -> Path:
    """
    Convert a RealEstate10K trajectory into the iPhone-dataset camera format.

    iPhoneDataset (camera_type="original") expects, for every frame
    <frame_name>, a JSON file at:
        <out_dir>/camera/<frame_name>.json
    containing:
        {
            "focal_length":      float,          # scalar (fx == fy assumed)
            "principal_point":   [cx, cy],
            "orientation":       (3, 3) R,       # w2c rotation (world -> cam)
            "position":          (3,)  o         # camera center in world coords
        }
    The loader reconstructs w2c as:
        w2c = [[orientation, -orientation @ position], [0, 0, 0, 1]]

    From the RealEstate10K w2c = [R | t] (world-to-camera):
        orientation = R
        position    = -R^T t        (camera center in world coordinates)

    Args:
        pose_file: path to RealEstate10K .txt pose file.
        out_dir:   output root (camera JSONs written to <out_dir>/camera/).
        seq_name:  sequence name (used to name frames "seq_name_00000" style,
                   matching the iPhone frame naming convention).

    Returns:
        Path to the output camera directory.
    """
    cams = read_realestate10k_poses(pose_file)
    cam_dir = Path(out_dir) / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(cams)
    pad = len(str(max(num_frames - 1, 0)))
    for i, cam in enumerate(cams):
        R = cam["w2c"][:3, :3]
        t = cam["w2c"][:3, 3]
        orientation = R
        position = -R.T @ t
        focal = (cam["fx"] + cam["fy"]) / 2.0  # iPhone assumes single focal length

        frame_name = f"{seq_name}_{i:0{pad}d}"
        cam_dict = {
            "focal_length": float(focal),
            "principal_point": [float(cam["cx"]), float(cam["cy"])],
            "orientation": orientation.tolist(),
            "position": position.tolist(),
        }
        with open(cam_dir / f"{frame_name}.json", "w") as f:
            json.dump(cam_dict, f, indent=2)

    print(f"[sharedCamTrajectory] RealEstate10K -> iPhone cameras: {cam_dir}")
    print(f"    frames : {num_frames}")
    print(f"    example: {cam_dir / frame_name}.json")
    return cam_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_intrinsics(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    assert len(vals) == 4, "intrinsics must be 'fx,fy,cx,cy'"
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared Plücker trajectory interface (generation <-> reconstruction)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- plucker-trajectory
    p1 = sub.add_parser("plucker-trajectory", help="Generate a trajectory + Plücker embedding.")
    p1.add_argument("--ref-w2c", type=str, default=None,
                    help="Path to a reference world-to-camera .npy (N,4,4) or (4,4).")
    p1.add_argument("--fx", type=float, default=700.0)
    p1.add_argument("--fy", type=float, default=700.0)
    p1.add_argument("--cx", type=float, default=320.0)
    p1.add_argument("--cy", type=float, default=240.0)
    p1.add_argument("--height", type=int, default=480)
    p1.add_argument("--width", type=int, default=720)
    p1.add_argument("--traj-type", type=str, default="spiral",
                    choices=["spiral", "orbit", "lemniscate", "arc", "wander", "fixed"])
    p1.add_argument("--num-frames", type=int, default=80)
    p1.add_argument("--radius", type=float, default=1.0)
    p1.add_argument("--out-dir", type=str, default="./data")
    p1.add_argument("--seq-name", type=str, default="my_scene")
    p1.add_argument("--save-plucker", type=str, default=None,
                    help="Optional .npy path to also save the (N,H,W,6) Plücker tensor.")

    # -- convert
    p2 = sub.add_parser("convert", help="Convert an existing trajectory to a camera file.")
    p2.add_argument("--traj-c2w", type=str, required=True, help="Path to (N,4,4) camera-to-world .npy.")
    p2.add_argument("--intrinsics", type=str, required=True, help="'fx,fy,cx,cy'")
    p2.add_argument("--img-shape", type=str, required=True, help="'h,w'")
    p2.add_argument("--out-dir", type=str, default="./data")
    p2.add_argument("--seq-name", type=str, default="my_scene")

    # -- realestate10k-to-iphone
    p2b = sub.add_parser(
        "realestate10k-to-iphone",
        help="Convert a RealEstate10K pose file into iPhone-style camera JSONs.",
    )
    p2b.add_argument("--pose-file", type=str, required=True,
                     help="Path to RealEstate10K .txt pose file.")
    p2b.add_argument("--out-dir", type=str, default="./data")
    p2b.add_argument("--seq-name", type=str, default="my_scene")

    # -- inspect
    p3 = sub.add_parser("inspect", help="Inspect a shared camera file.")
    p3.add_argument("--cam-file", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "plucker-trajectory":
        # reference camera (optional)
        ref_w2c = None
        if args.ref_w2c:
            arr = np.load(args.ref_w2c)
            if arr.ndim == 3:
                arr = arr[0]
            ref_w2c = np.asarray(arr, dtype=np.float64)
            if ref_w2c.shape[0] != 4 or ref_w2c.shape[1] != 4:
                raise ValueError("ref-w2c must be (4,4) or (N,4,4).")

        traj_w2c = generate_trajectory(
            args.traj_type,
            args.num_frames,
            ref_w2c=ref_w2c,
            radius=args.radius,
        )
        # invert w2c -> c2w
        traj_c2w = np.linalg.inv(traj_w2c)
        intrinsics = (args.fx, args.fy, args.cx, args.cy)

        save_camera_file(
            traj_c2w,
            intrinsics,
            (args.height, args.width),
            None,
            args.out_dir,
            args.seq_name,
        )

        # optionally compute + save the Plücker trajectory (generation side)
        if args.save_plucker:
            K = np.array(
                [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                dtype=np.float64,
            )
            plucker = plucker_for_trajectory(traj_w2c, K, args.height, args.width)
            Path(args.save_plucker).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_plucker, plucker)
            print(f"[sharedCamTrajectory] saved Plücker tensor: {args.save_plucker} "
                  f"shape={plucker.shape}")

    elif args.mode == "convert":
        traj_c2w = np.load(args.traj_c2w)
        assert traj_c2w.ndim == 3 and traj_c2w.shape[1:] == (4, 4), \
            "traj-c2w must be (N, 4, 4)."
        h, w = [int(x) for x in args.img_shape.split(",")]
        intrinsics = parse_intrinsics(args.intrinsics)
        save_camera_file(
            traj_c2w, intrinsics, (h, w), None, args.out_dir, args.seq_name
        )

    elif args.mode == "realestate10k-to-iphone":
        cam_dir = realestate10k_to_iphone(
            args.pose_file, args.out_dir, args.seq_name
        )
        # also save a droid_recon-style shared camera file for completeness
        cams = read_realestate10k_poses(args.pose_file)
        traj_w2c = np.stack([c["w2c"] for c in cams], axis=0)
        traj_c2w = np.linalg.inv(traj_w2c)
        img_shape = (cams[0]["H"], cams[0]["W"])
        intrinsics = (cams[0]["fx"], cams[0]["fy"], cams[0]["cx"], cams[0]["cy"])
        save_camera_file(
            traj_c2w, intrinsics, img_shape, None, args.out_dir, args.seq_name
        )

    elif args.mode == "inspect":
        inspect_camera_file(args.cam_file)


if __name__ == "__main__":
    main()
