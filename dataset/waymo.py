import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import json
import cv2
import torch.nn.functional as F
import random
import copy
from .typing import *
from .utils import inv, geotrf, rescale_image_depthmap_custom, depthmap_to_absolute_camera_coordinates


class WaymoDataset(Dataset):
    def __init__(self, scene_root='data/waymo', is_train=True, test_interval=1, cfg=None):
        self.scene_root = scene_root
        self.samples = []
        scene_names = os.listdir(scene_root)

        trip_segment_id_list = []
        flag_idx = -1
        for scene in scene_names:
            flag_idx += 1
            frame_names = os.listdir(os.path.join(scene_root, scene, 'images'))
            filter_frames = [tmp for tmp in frame_names if tmp.endswith(".png")]
            frame_names = [tmp.split("_")[0] for tmp in filter_frames]
            frame_names = set(frame_names)
            frame_names = list(frame_names)
            frame_names.sort()
            
            for frame_name in frame_names:
                if cfg.get('use_frames', -1) > 0:
                    if int(frame_name) > cfg.use_frames:
                        continue
                self.samples.append((scene, frame_name))
                trip_segment_id_list.append(flag_idx)
        
        self.is_train = is_train
        if self.is_train == False:
            self.samples = self.samples[::test_interval]
        sample_dict = {}
        for i in range(len(self.samples)):
            scene, frame_name = self.samples[i]
            if scene not in sample_dict:
                sample_dict[scene] = {}
            sample_dict[scene][frame_name] = i
        self.sample_dict = sample_dict
        
        self.future_num = cfg.get('future_num', 5)
        self.frame_num = cfg.get('frame_num', 1)
        self.resolution = cfg.get('resolution')
        self.cfg = cfg
        
        self.seq_flag, self.group_idx_to_sample_idxs = self.set_sequence_group_flag(trip_segment_id_list)
    
    
    def __len__(self):
        return len(self.samples)
    
    def load_instance(self, idx, select_cam=None):
        scene, frame_name = self.samples[idx]
        images = []
        camera_poses = []
        intrinsics_list = []
        camera2lidar_list = []
        single_frame_depths, pts3ds_single, valid_masks_single = [], [], []
        image_path_list = []
        sky_mask_list = []
        
        # use for 3D GS, actually use lidar as world coords 
        input_c2ws, input_w2cs = [], []
        dynamics_region_list = []
        with open(os.path.join(self.scene_root, scene, 'dynamics', 'dynamic_infos.json'), 'r') as f:
            dynamic_info = json.load(f)['track_id_infos'][str(int(frame_name))]
        for i in range(1, 6):
            if select_cam is not None and i not in select_cam:
                continue
            image_path = os.path.join(self.scene_root, scene, "images", f"{frame_name}_{i-1}.png")
            depthmap_path = os.path.join(self.scene_root, scene, f"{frame_name[1:]}_{i}.exr")
            calib_path = os.path.join(self.scene_root, scene, f"{frame_name[1:]}_{i}.npz")
            dynamics_path = os.path.join(self.scene_root, scene, 'dynamics')   
            with np.load(os.path.join(dynamics_path, f"dynamic_mask_{int(frame_name)}_{i}.npz")) as loaded_data:
                dynamics_mask = loaded_data['data']
            dynamics_region = copy.deepcopy(dynamics_mask)
            for track_id in dynamic_info[str(i)]:
                track_speed = dynamic_info[str(i)][track_id]['speed']
                track_speed = np.linalg.norm(track_speed)
                if track_speed < 0.2:
                    dynamics_region[dynamics_mask == int(track_id)] = 0
            dynamics_region = dynamics_region > 0
            
            image = imread_cv2(image_path)
            moge_depth_mask = os.path.join(self.scene_root, scene, f"{frame_name[1:]}_{i}_moge_mask.png")
            sky_mask = cv2.imread(moge_depth_mask, cv2.IMREAD_GRAYSCALE)>0 
            depthmap = imread_cv2(depthmap_path)
            calib = np.load(calib_path)

            camera2lidar = np.float32(calib["cam2lidar"])
            intrinsics = np.float32(calib["intrinsics"])
            camera_pose = np.float32(calib["cam2world"])
            single_frame_depth = depthmap.copy()
            resolution = self.resolution
            image, _, tmp_intrinsics = rescale_image_depthmap_custom(image, None, intrinsics, resolution, force=True)
            W, H = resolution
            single_frame_depth_resize = np.zeros((H, W), dtype=np.float32)
            y, x = np.where(single_frame_depth > 0)
            pos2d = np.stack([x,y], axis=-1)  # (N, 2)
            depth_value = single_frame_depth[y, x]
            pos2d = geotrf(tmp_intrinsics @ inv(intrinsics), pos2d).round().astype(np.int16)
            x, y = pos2d.T
            single_frame_depth_resize[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = depth_value
            pts3d_single, valid_mask_single = depthmap_to_absolute_camera_coordinates(single_frame_depth_resize, tmp_intrinsics.copy(), camera_pose)
            # process moge depthmap
            sky_mask = cv2.resize(sky_mask.astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST)
            # process dynamic mask
            dynamics_region = cv2.resize(dynamics_region.astype(np.float32), resolution, interpolation=cv2.INTER_NEAREST)
            # final intrinsics
            intrinsics = tmp_intrinsics

            images.append(image)
            camera_poses.append(camera_pose)
            intrinsics_list.append(intrinsics)
            camera2lidar_list.append(camera2lidar)
            image_path_list.append(image_path)
            single_frame_depths.append(single_frame_depth_resize)
            pts3ds_single.append(pts3d_single)
            valid_masks_single.append(valid_mask_single)
            input_c2ws.append(camera2lidar)
            input_w2cs.append(inv(camera2lidar))
            sky_mask_list.append(sky_mask)
            dynamics_region_list.append(dynamics_region)
        
        to_tensor_transform = transforms.ToTensor()
        tensor_list = []
        for image in images:
            tensor_list.append(to_tensor_transform(image))
        images = torch.stack(tensor_list, dim=0)
        camera_poses = np.stack(camera_poses, axis=0)
        intrinsics_list = np.stack(intrinsics_list, axis=0)
        camera2lidar_list = np.stack(camera2lidar_list, axis=0)
        single_frame_depths = np.stack(single_frame_depths, axis=0)
        pts3ds_single = np.stack(pts3ds_single, axis=0)
        valid_masks_single = np.stack(valid_masks_single, axis=0)
        input_c2ws = torch.as_tensor(np.stack(input_c2ws, axis=0), dtype=torch.float32)
        input_w2cs = torch.as_tensor(np.stack(input_w2cs, axis=0), dtype=torch.float32)
        sky_mask_list = np.stack(sky_mask_list, axis=0)
        

        return {
            'images': images,
            'scene': scene,
            'frame': frame_name,
            'camera_pose': camera_poses,
            'intrinsics': intrinsics_list,
            'camera2lidar': camera2lidar_list,
            'image_path': image_path_list,
            'single_depthmaps': single_frame_depths,
            'single_pts3d': pts3ds_single,
            'single_pts3d_valid_mask': valid_masks_single,
            'input_c2ws': input_c2ws,
            'input_w2cs': input_w2cs,
            'sky_mask': sky_mask_list,
            'dynamics_region': np.stack(dynamics_region_list, axis=0),
        }
    
    def __getitem__(self, idx):
        scene, frame = self.samples[idx]

        current_dict = self.load_instance(idx)
        # for 3D GS
        input_c2ws = current_dict['input_c2ws']
        input_w2cs = current_dict['input_w2cs']
        input_cks = torch.as_tensor(current_dict['intrinsics'], dtype=torch.float32)
        input_fxs, input_fys, input_cxs, input_cys = input_cks[:, 0, 0], input_cks[:, 1, 1], input_cks[:, 0, 2], input_cks[:, 1, 2]
        # compute image fovs and pixel directions
        input_fovxs, input_fovys = [], []
        input_directions = []
        for fx, fy, cx, cy in zip(input_fxs, input_fys, input_cxs, input_cys):
            direction = get_ray_directions(self.resolution[1], self.resolution[0], focal=[fx, fy], principal=[cx, cy])
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            input_fovxs.append(fovx)
            input_fovys.append(fovy)
            input_directions.append(direction)
        input_fovxs = torch.as_tensor(input_fovxs, dtype=torch.float32)
        input_fovys = torch.as_tensor(input_fovys, dtype=torch.float32)
        input_directions = torch.stack(input_directions)
        input_rays_o, input_rays_d = get_rays(
            input_directions, input_c2ws, keepdim=True, normalize=False)
        
        # ======= Render views from non-key frames for rendering losses ====== #
        input_camera_poses = torch.as_tensor(current_dict['camera_pose'], dtype=torch.float32)
        scene, frame_name = current_dict['scene'], current_dict['frame']
        frames_list = self.sample_dict[scene].keys()
        frames_list = [int(tmp) for tmp in frames_list]
        output_imgs, output_c2ws = [], []
        output_cks, sky_mask = [], []
        dynamics_region = []
        for i in range(1, self.future_num+1):
            after_frame = int(frame_name) + i
            after_frame = min(after_frame, max(frames_list))
            after_frame = str(after_frame).zfill(len(frame_name))
            after_index = self.sample_dict[scene][after_frame]
            select_cam = []
            for j in range(self.frame_num):
                cam_index = j+1
                select_cam.append(cam_index)
            after_dict = self.load_instance(after_index, select_cam=select_cam)
            tmp = []
            for j, cam_index in enumerate(select_cam):
                after_c2ws = input_c2ws[cam_index-1:cam_index] @ inv(input_camera_poses[cam_index-1:cam_index]) @ after_dict['camera_pose'][j:j+1]
                tmp.append(after_c2ws)
            after_c2ws = torch.cat(tmp, dim=0)
            output_imgs.append(after_dict['images'])
            output_c2ws.append(after_c2ws)
            output_cks.append(after_dict['intrinsics'])
            sky_mask.append(after_dict['sky_mask'])
            dynamics_region.append(after_dict['dynamics_region'])
        output_imgs = torch.cat(output_imgs, dim=0)
        output_c2ws = torch.cat(output_c2ws, dim=0)
        sky_mask = np.concatenate(sky_mask, axis=0)
        dynamics_region = np.concatenate(dynamics_region, axis=0)
        output_cks = np.concatenate(output_cks, axis=0)
        output_cks = torch.as_tensor(output_cks, dtype=torch.float32)
        output_fxs, output_fys, output_cxs, output_cys = output_cks[:, 0, 0], output_cks[:, 1, 1], output_cks[:, 0, 2], output_cks[:, 1, 2]
        # compute image fovs and pixel directions
        output_fovxs, output_fovys = [], []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            output_fovxs.append(fovx)
            output_fovys.append(fovy)
        output_fovxs = torch.as_tensor(output_fovxs, dtype=torch.float32)
        output_fovys = torch.as_tensor(output_fovys, dtype=torch.float32)
            
        # add input data to output
        current_dynamics_region = np.zeros_like(current_dict['dynamics_region'])
        output_dynamics_region = np.concatenate([dynamics_region, current_dynamics_region], axis=0)
        output_imgs = torch.cat([output_imgs, current_dict['images']], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_fovxs = torch.cat([output_fovxs, input_fovxs], dim=0)
        output_fovys = torch.cat([output_fovys, input_fovys], dim=0)
        output_fxs = torch.cat([output_fxs, input_fxs], dim=0)
        output_fys = torch.cat([output_fys, input_fys], dim=0)
        output_cxs = torch.cat([output_cxs, input_cxs], dim=0)
        output_cys = torch.cat([output_cys, input_cys], dim=0)
        output_directions = []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            direction = get_ray_directions(self.resolution[1], self.resolution[0], focal=[fx, fy], principal=[cx, cy])
            output_directions.append(direction)
        output_directions = torch.stack(output_directions)
        output_rays_o, output_rays_d = get_rays(
                    output_directions, output_c2ws, keepdim=True, normalize=False)
        
        sky_mask = np.concatenate([sky_mask, current_dict['sky_mask']], axis=0)
        sky_mask = torch.as_tensor(sky_mask, dtype=torch.bool)
        output_dict = {"rgb": output_imgs,
                "c2w": output_c2ws, "fovx": output_fovxs, "fovy": output_fovys,
                "rays_o": output_rays_o, "rays_d": output_rays_d, "sky_mask": sky_mask,
                "dynamics_region": output_dynamics_region, "intrinsics": torch.cat([output_cks, input_cks], dim=0)}
        input_dict_pix = {"ck": input_cks, "c2w": input_c2ws,
                    "cx": input_cxs, "cy": input_cys, "fx": input_fxs, "fy": input_fys,
                    "rays_o": input_rays_o, "rays_d": input_rays_d}
                
        current_dict['output_dict_gs'] = output_dict
        current_dict['input_dict_gs'] = input_dict_pix
        
        return current_dict

    def set_sequence_group_flag(self, trip_segment_id_list):
        """Set each sequence to be a different group."""
        samples_num = len(trip_segment_id_list)
        seq_flag = np.array(trip_segment_id_list, dtype=np.int64)
        num_frame_each_split_group = 20
        if self.is_train == False:
            num_frame_each_split_group = 20
        group_idx_to_sample_idxs = dict()

        if num_frame_each_split_group == -1:
            # maintain the original sequence
            return seq_flag, group_idx_to_sample_idxs

        if num_frame_each_split_group == 1:
            # each frame is a different group
            group_idx_to_sample_idxs = {group_idx: [group_idx] for group_idx in range(samples_num)}
            return np.array(range(samples_num), dtype=np.int64), group_idx_to_sample_idxs

        # split each trip into multiple subgroups
        seq_flag = np.array(seq_flag, dtype=np.int64)
        bin_counts = np.bincount(seq_flag)
        new_flags = []
        curr_new_flag = 0
        for idx in range(len(bin_counts)):
            num_frame = bin_counts[idx]

            if num_frame < num_frame_each_split_group:
                group_idx_to_sample_idxs[curr_new_flag] = list(range(len(new_flags), len(new_flags) + num_frame))
                new_flags.extend([curr_new_flag] * num_frame)

                curr_new_flag += 1
                continue

            for frame_idx in range(0, num_frame, num_frame_each_split_group):
                cnt = min(num_frame_each_split_group, num_frame - frame_idx)
                group_idx_to_sample_idxs[curr_new_flag] = list(range(len(new_flags), len(new_flags) + cnt))
                new_flags.extend([curr_new_flag] * cnt)
                curr_new_flag += 1

        assert len(new_flags) == len(seq_flag)
        seq_flag = np.array(new_flags, dtype=np.int64)

        print(f'Complete setting sequence group flag, num_frame_each_split_group={num_frame_each_split_group}')
        return seq_flag, group_idx_to_sample_idxs


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    # directions: Float[Tensor, "H W 3"] = torch.stack(
    #     [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    # )
    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=True,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


