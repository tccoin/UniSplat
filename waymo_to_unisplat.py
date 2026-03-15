#!/usr/bin/env python3
"""Convert STORM-preprocessed Waymo data to UniSplat format.

Input layout (STORM):
  <storm_dir>/training/<seq_id>/
    images_4/<frame>_<cam>.jpg       # 320x480 (cam 0-4)
    cam_to_world/<frame>_<cam>.txt   # 4x4 float64, cam2world
    cam_to_ego/<cam>.txt             # 4x4 float64, cam2vehicle (= cam2lidar in Waymo)
    intrinsics/<cam>.txt             # fx fy cx cy at ORIGINAL resolution
    depth_flows_4/<frame>_<cam>.npy  # (H, W, 4): [depth_m, flow_x, flow_y, valid]
    sky_masks/<frame>_<cam>.png      # binary sky mask
    dynamic_masks/<frame>_<cam>.png  # binary dynamic mask

Output layout (UniSplat):
  <out_dir>/<scene_name>/
    images/<frame>_<cam_idx>.png              # RGB PNG, cam 0-4, frame 4-digit
    <frame[1:]>_<cam_1-5>.exr                # float32 EXR depth (metres)
    <frame[1:]>_<cam_1-5>.npz                # {cam2lidar, intrinsics(3x3), cam2world}
    <frame[1:]>_<cam_1-5>_moge_mask.png      # sky mask as binary PNG
    dynamics/dynamic_infos.json              # stub {}
    dynamics/dynamic_mask_<frame>_<cam>.npz  # dynamic mask

Usage:
  python scripts/waymo_to_unisplat.py \\
      --storm_dir data/processed/storm/waymo/training \\
      --out_dir   data/processed/unisplat/waymo \\
      --seq_items "002:98:198;031:0:198;036:0:198"
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

# Waymo segment name map: scene_id (zero-padded 3-digit) -> segment name
SCENE_ID_TO_SEGMENT = {
    "002": "segment-1005081002024129653_5313_150_5333_150_with_camera_labels",
    "031": "segment-10588771936253546636_2300_000_2320_000_with_camera_labels",
    "036": "segment-10676267326664322837_311_180_331_180_with_camera_labels",
}



CV_TO_FLU = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

def save_exr(path, depth_arr):
    """Save float32 2D array as single-channel EXR via cv2 (OPENCV_IO_ENABLE_OPENEXR=1 required)."""
    import os, cv2
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    cv2.imwrite(path, depth_arr.astype(np.float32))


def load_4x4(path):
    return np.loadtxt(path, dtype=np.float64)


def convert_sequence(storm_seq_dir, out_scene_dir, start_frame, end_frame):
    """Convert one STORM sequence folder to UniSplat format."""
    os.makedirs(os.path.join(out_scene_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_scene_dir, "dynamics"), exist_ok=True)

    # dynamic_infos.json written after frame list is known (below)
    dyn_info_path = os.path.join(out_scene_dir, "dynamics", "dynamic_infos.json")

    # Load per-cam static calibration (intrinsics + cam2lidar)
    cam_calib = {}
    for cam in range(5):
        intrinsics_path = os.path.join(storm_seq_dir, "intrinsics", f"{cam}.txt")
        cam2ego_path = os.path.join(storm_seq_dir, "cam_to_ego", f"{cam}.txt")
        raw_intr = np.loadtxt(intrinsics_path, dtype=np.float64).flatten()
        fx, fy, cx, cy = raw_intr[0], raw_intr[1], raw_intr[2], raw_intr[3]
        # intrinsics/<cam>.txt stores values at ORIGINAL resolution
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cam2lidar = load_4x4(cam2ego_path) @ CV_TO_FLU
        cam_calib[cam] = {"K": K, "cam2lidar": cam2lidar}

    # Enumerate frames
    img4_dir = os.path.join(storm_seq_dir, "images_4")
    all_frames = sorted({f.split("_")[0] for f in os.listdir(img4_dir) if f.endswith(".jpg")})
    all_frames = [f for f in all_frames if start_frame <= int(f) <= end_frame]

    # Build dynamic_infos.json stub with proper per-frame/per-cam structure
    if not os.path.exists(dyn_info_path):
        track_id_infos = {}
        for frame_str in all_frames:
            frame_int = int(frame_str)
            track_id_infos[str(frame_int)] = {str(c): {} for c in range(1, 6)}
        dyn_info = {"track_id_infos": track_id_infos, "track_id_dict": {}}
        with open(dyn_info_path, "w") as f:
            json.dump(dyn_info, f)

    for frame_str in tqdm(all_frames, desc=os.path.basename(out_scene_dir)):
        frame_int = int(frame_str)
        # UniSplat uses 4-digit zero-padded frame names in images/
        frame4 = f"{frame_int:04d}"
        # EXR/NPZ/mask use frame4[1:] = last 3 digits (matches UniSplat dataset/waymo.py)
        exr_prefix = frame4[1:]

        for cam in range(5):
            # 1. Image JPG -> PNG
            src_img = os.path.join(img4_dir, f"{frame_str}_{cam}.jpg")
            dst_img = os.path.join(out_scene_dir, "images", f"{frame4}_{cam}.png")
            if not os.path.exists(dst_img) and os.path.exists(src_img):
                Image.open(src_img).convert("RGB").save(dst_img)

            # 2. cam2world (per frame per cam)
            c2w_path = os.path.join(storm_seq_dir, "cam_to_world", f"{frame_str}_{cam}.txt")
            if not os.path.exists(c2w_path):
                continue
            cam2world = load_4x4(c2w_path) @ CV_TO_FLU

            # 3. NPZ: cam2lidar (static) + intrinsics (static) + cam2world (per-frame)
            npz_path = os.path.join(out_scene_dir, f"{exr_prefix}_{cam+1}.npz")
            if not os.path.exists(npz_path):
                np.savez_compressed(
                    npz_path,
                    cam2lidar=cam_calib[cam]["cam2lidar"].astype(np.float32),
                    intrinsics=cam_calib[cam]["K"].astype(np.float32),
                    cam2world=cam2world.astype(np.float32),
                )

            # 4. Depth EXR
            exr_path = os.path.join(out_scene_dir, f"{exr_prefix}_{cam+1}.exr")
            if not os.path.exists(exr_path):
                depth_npy = os.path.join(
                    storm_seq_dir, "depth_flows_4", f"{frame_str}_{cam}.npy"
                )
                if os.path.exists(depth_npy):
                    df = np.load(depth_npy)
                    depth = df[..., 0].astype(np.float32)
                    save_exr(exr_path, depth)

            # 5. Sky mask (moge_mask)
            mask_dst = os.path.join(out_scene_dir, f"{exr_prefix}_{cam+1}_moge_mask.png")
            if not os.path.exists(mask_dst):
                sky_src = os.path.join(storm_seq_dir, "sky_masks", f"{frame_str}_{cam}.png")
                if os.path.exists(sky_src):
                    shutil.copy2(sky_src, mask_dst)

            # 6. Dynamic mask NPZ
            dyn_dst = os.path.join(
                out_scene_dir, "dynamics", f"dynamic_mask_{frame_int}_{cam+1}.npz"
            )
            if not os.path.exists(dyn_dst):
                dyn_src = os.path.join(
                    storm_seq_dir, "dynamic_masks", f"{frame_str}_{cam}.png"
                )
                if os.path.exists(dyn_src):
                    mask_arr = np.array(Image.open(dyn_src).convert("L"))
                    np.savez_compressed(dyn_dst, data=mask_arr)


def parse_seq_items(seq_items_str):
    """Parse '002:98:198;031:0:198' -> [(seq_id, start, end), ...]."""
    result = []
    for item in seq_items_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        seq_id = parts[0].zfill(3)
        start = int(parts[1]) if len(parts) > 1 else 0
        end = int(parts[2]) if len(parts) > 2 else 999999
        result.append((seq_id, start, end))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storm_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_items", required=True)
    args = parser.parse_args()

    for seq_id, start, end in parse_seq_items(args.seq_items):
        storm_seq_dir = os.path.join(args.storm_dir, seq_id)
        if not os.path.isdir(storm_seq_dir):
            print(f"[SKIP] not found: {storm_seq_dir}")
            continue
        segment_name = SCENE_ID_TO_SEGMENT.get(seq_id, seq_id)
        out_scene_dir = os.path.join(args.out_dir, segment_name)
        print(f"\n[{seq_id}] -> {out_scene_dir}  frames {start}-{end}")
        convert_sequence(storm_seq_dir, out_scene_dir, start, end)
        print(f"[{seq_id}] Done.")


if __name__ == "__main__":
    main()
