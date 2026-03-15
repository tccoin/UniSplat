#!/usr/bin/env python3
import argparse
import copy
import math
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import sys

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
UNISPLAT_DIR = os.path.join(REPO_ROOT, 'submodules/UniSplat')
sys.path.insert(0, UNISPLAT_DIR)

from dataset.waymo import WaymoDataset
import model.gaussian_head as gaussian_head_class
from pi3.models.pi3 import Pi3

SCENE_ID_TO_SEGMENT = {
    '002': 'segment-1005081002024129653_5313_150_5333_150_with_camera_labels',
    '031': 'segment-10588771936253546636_2300_000_2320_000_with_camera_labels',
    '036': 'segment-10676267326664322837_311_180_331_180_with_camera_labels',
}
SEGMENT_TO_SEQ = {v: k for k, v in SCENE_ID_TO_SEGMENT.items()}
NUM_CAMS = 5
FPS = 10


def parse_seq_items(s):
    out = {}
    for item in s.split(';'):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        seq = parts[0].zfill(3)
        start = int(parts[1]) if len(parts) > 1 else 0
        end = int(parts[2]) if len(parts) > 2 else 999999
        seg = SCENE_ID_TO_SEGMENT.get(seq, seq)
        out[seg] = {'seq': seq, 'start': start, 'end': end}
    return out


def elevate_camtoworlds(c2w: torch.Tensor, height: float, tilt_deg: float) -> torch.Tensor:
    c2w = c2w.clone()
    c2w[..., 2, 3] += height
    angle = math.radians(tilt_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    Ry = torch.tensor(
        [[cos_a, 0, sin_a, 0],
         [0, 1, 0, 0],
         [-sin_a, 0, cos_a, 0],
         [0, 0, 0, 1]], dtype=c2w.dtype, device=c2w.device)
    return c2w @ Ry


def save_img_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def move_top_level_batch(batch, device):
    for key in list(batch.keys()):
        if key not in ['input_dict_gs', 'output_dict_gs'] and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def flush_scene(scene_name, buffers, out_dir):
    seq = SEGMENT_TO_SEQ.get(scene_name, scene_name)
    out_seq_dir = os.path.join(out_dir, seq)
    vid_dir = os.path.join(out_seq_dir, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    print(f"\n[{seq}] Saving videos to {vid_dir} ...")
    for cam in range(NUM_CAMS):
        imageio.mimwrite(os.path.join(vid_dir, f'cam{cam}_gt_rgb.mp4'), buffers['gt'][cam], fps=FPS, macro_block_size=None)
        imageio.mimwrite(os.path.join(vid_dir, f'cam{cam}_rgb.mp4'), buffers['rgb'][cam], fps=FPS, macro_block_size=None)
        imageio.mimwrite(os.path.join(vid_dir, f'cam{cam}_novel_rgb.mp4'), buffers['novel'][cam], fps=FPS, macro_block_size=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seq_items', required=True)
    parser.add_argument('--elevated_height', type=float, default=2.0)
    parser.add_argument('--elevated_tilt_deg', type=float, default=15.0)
    args = parser.parse_args()

    seq_map = parse_seq_items(args.seq_items)
    cfg = OmegaConf.load(args.config)
    dataset = WaymoDataset(scene_root=args.data_dir, is_train=False, cfg=cfg.Dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = Pi3()
    model_name = cfg.Model.Gaussian_head.Name
    model_class = getattr(gaussian_head_class, model_name)
    model.gaussian_head = model_class(dim_in=2048, cfg=cfg.Model.Gaussian_head)
    model.gaussian_head.image_backbone.mask_token = None
    weight = load_file(args.ckpt, device='cpu')
    model.load_state_dict(weight, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    current_scene = None
    buffers = None

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='unisplat-eval'):
            scene = batch['scene'][0]
            frame = batch['frame'][0]
            if scene not in seq_map:
                continue
            frame_int = int(frame)
            if not (seq_map[scene]['start'] <= frame_int <= seq_map[scene]['end']):
                continue

            if current_scene != scene:
                if current_scene is not None and buffers is not None:
                    flush_scene(current_scene, buffers, args.out_dir)
                QueueClass = type(model.gaussian_head.history_queue)
                model.gaussian_head.history_queue = QueueClass()
                current_scene = scene
                buffers = {
                    'gt': [[] for _ in range(NUM_CAMS)],
                    'rgb': [[] for _ in range(NUM_CAMS)],
                    'novel': [[] for _ in range(NUM_CAMS)],
                }

            batch = move_top_level_batch(batch, device)
            images = batch['images'].to(device)

            with autocast(device_type='cuda', dtype=amp_dtype):
                res = model(images)

            # ---- Step 1: Standard Recon Render (Future + Recon) ----
            with autocast(device_type='cuda', enabled=False):
                idgs = batch['input_dict_gs']
                idgs['sky_mask'] = batch['sky_mask'].to(images.dtype).to(device)
                idgs['single_depthmaps'] = batch['single_depthmaps'].to(device)
                batch['input_dict_gs']['intrinsics'] = batch['intrinsics'].to(device)
                batch['input_dict_gs']['camera2lidar'] = batch['camera2lidar'].to(device)
                batch['input_dict_gs']['dynamics_region'] = batch['dynamics_region'].to(device)
                res['scene'] = batch['scene']
                res['frame'] = batch['frame']
                lidar2world = batch['camera_pose'] @ batch['camera2lidar'].inverse()
                res['lidar2world'] = lidar2world
                
                # Render 10 views (standard demo behavior)
                render_pkg_recon = model.gaussian_head(res, images, 5, input_dict_gs=idgs, output_dict_gs=batch['output_dict_gs'])[0]

            rgb_pred = render_pkg_recon['image'] # (10, 3, H, W)
            rgb_gt = batch['output_dict_gs']['rgb'].to(rgb_pred.device, rgb_pred.dtype)[0]
            for cam in range(NUM_CAMS):
                buffers['gt'][cam].append(save_img_uint8(rgb_gt[cam + 5]))
                buffers['rgb'][cam].append(save_img_uint8(rgb_pred[cam + 5]))

            # Save Gaussians
            gs_dir = os.path.join(args.out_dir, seq_map[scene]['seq'], 'gaussians')
            os.makedirs(gs_dir, exist_ok=True)
            gs_list = model.gaussian_head.history_queue.gaussians
            if gs_list and gs_list[0] is not None:
                torch.save(gs_list[0].cpu(), os.path.join(gs_dir, f'{frame}.pth'))

            # ---- Step 2: Elevated Novel Render (Separately) ----
            # To save memory, we render elevated views in a separate call
            # and temporarily disable history caching to avoid redundant updates.
            with autocast(device_type='cuda', enabled=False):
                queue = model.gaussian_head.history_queue
                # Snapshot state
                saved_state = {field: getattr(queue, field) for field in queue.cache_fields + ['scenes', 'frames']}
                # Deepcopy list contents only for what's modified (if any), 
                # but scenes/frames/gaussians are what get updated.
                
                elev_odgs = {}
                for k, v in batch['output_dict_gs'].items():
                    # We only care about current cams elevated
                    if isinstance(v, torch.Tensor):
                        elev_odgs[k] = v[:, 5:10].clone().to(device)
                    else:
                        elev_odgs[k] = v
                
                elev_odgs['c2w'] = elevate_camtoworlds(elev_odgs['c2w'], args.elevated_height, args.elevated_tilt_deg)
                
                # Manual cache clear before heavy op
                torch.cuda.empty_cache()
                
                render_pkg_elev = model.gaussian_head(res, images, 5, input_dict_gs=idgs, output_dict_gs=elev_odgs)[0]
                
                # Restore history queue state to previous standard recon state
                for field, value in saved_state.items():
                    setattr(queue, field, value)
            
            rgb_elev = render_pkg_elev['image'] # (5, 3, H, W)
            for cam in range(NUM_CAMS):
                buffers['novel'][cam].append(save_img_uint8(rgb_elev[cam]))

            # Final cleanup per frame
            del render_pkg_recon, render_pkg_elev, rgb_pred, rgb_elev
            torch.cuda.empty_cache()

        if current_scene is not None and buffers is not None:
            flush_scene(current_scene, buffers, args.out_dir)

if __name__ == '__main__':
    main()
