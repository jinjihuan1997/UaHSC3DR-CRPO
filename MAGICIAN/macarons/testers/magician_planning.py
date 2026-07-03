import os
import sys
import gc
import shutil
import csv
import hashlib
import json
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
from ..utility.gaussian_utils import CamerasWrapper, convert_camera_from_pytorch3d_to_gs
from ..utility.magician_utils import *
import numpy as np
import trimesh
import lmdb
import time

# ==================== RaDe-GS Integration ====================
RADE_GS_PATH = os.path.join(os.path.dirname(__file__), "../../RaDe-GS")
if RADE_GS_PATH not in sys.path:
    sys.path.insert(0, RADE_GS_PATH)


def _sync_if_cuda(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _profile_now(device):
    _sync_if_cuda(device)
    return time.perf_counter()


def _frame_communication_bytes(camera, frame_idx):
    frames_dir = camera.save_dir_path
    raw_pt_bytes = 0
    rgb_png_bytes = 0
    if frames_dir is not None:
        frame_path = os.path.join(frames_dir, f"{frame_idx}.pt")
        if os.path.exists(frame_path):
            raw_pt_bytes = os.path.getsize(frame_path)

        imgs_dir = os.path.join(os.path.dirname(frames_dir), "imgs")
        png_path = os.path.join(imgs_dir, f"{frame_idx}.png")
        if os.path.exists(png_path):
            rgb_png_bytes = os.path.getsize(png_path)

    n_pixels = int(camera.image_height * camera.image_width)
    depth_uint16_bytes = 2 * n_pixels
    mask_bitpacked_bytes = (n_pixels + 7) // 8
    pose_intrinsics_bytes = (12 + 9 + 3) * 4
    lossless_sensor_est_bytes = (
        rgb_png_bytes + depth_uint16_bytes + mask_bitpacked_bytes + pose_intrinsics_bytes
    )

    return {
        'raw_pt_bytes': int(raw_pt_bytes),
        'rgb_png_bytes': int(rgb_png_bytes),
        'depth_uint16_bytes': int(depth_uint16_bytes),
        'mask_bitpacked_bytes': int(mask_bitpacked_bytes),
        'pose_intrinsics_bytes': int(pose_intrinsics_bytes),
        'lossless_sensor_est_bytes': int(lossless_sensor_est_bytes),
    }


def _resize_frame_for_planning(images, depth, planning_height, planning_width):
    images_chw = images.permute(0, 3, 1, 2)
    depth_chw = depth.permute(0, 3, 1, 2)
    mask_chw = (depth > -1).float().permute(0, 3, 1, 2)

    low_images = torch.nn.functional.interpolate(
        images_chw, size=(planning_height, planning_width), mode="area"
    ).permute(0, 2, 3, 1).contiguous()
    low_depth = torch.nn.functional.interpolate(
        depth_chw, size=(planning_height, planning_width), mode="nearest"
    ).permute(0, 2, 3, 1).contiguous()
    low_mask = torch.nn.functional.interpolate(
        mask_chw, size=(planning_height, planning_width), mode="nearest"
    ).permute(0, 2, 3, 1).bool().contiguous()

    return low_images, low_depth, low_mask


def _pose_idx_key(pose_idx):
    pose_idx = torch.as_tensor(pose_idx).long().detach().cpu().view(-1)
    return tuple(int(v) for v in pose_idx.tolist())


def _occupied_position_keys(occupied_pose_data):
    if occupied_pose_data is None:
        return set()

    occupied_keys = set()
    x_indices = occupied_pose_data.get('X_idx', [])
    occupied = occupied_pose_data.get('occupied', [])
    for x_idx, is_occupied in zip(x_indices, occupied):
        if bool(torch.as_tensor(is_occupied).item()):
            occupied_keys.add(_pose_idx_key(x_idx)[:3])
    return occupied_keys


def _ensure_trajectory_memory_dirs(training_frames_path):
    trajectory_dir = os.path.dirname(training_frames_path)
    for subdir in ('frames', 'frames_highres', 'imgs', 'depths', 'occupancy'):
        os.makedirs(os.path.join(trajectory_dir, subdir), exist_ok=True)

    poses_dir = os.path.join(os.path.dirname(trajectory_dir), 'poses')
    os.makedirs(poses_dir, exist_ok=True)


def _build_scene_start_positions(settings, occupied_pose_data, requested_count, device, scene_name):
    base_starts = settings.camera.start_positions.long().to(device).view(-1, 5)
    if requested_count is None or int(requested_count) <= 0:
        return base_starts

    requested_count = int(requested_count)
    if requested_count <= len(base_starts):
        print(f"Limiting trajectories for {scene_name} to {requested_count} "
              f"of {len(base_starts)} configured start positions.")
        return base_starts[:requested_count]

    existing_keys = {_pose_idx_key(pose_idx) for pose_idx in base_starts}
    occupied_position_keys = _occupied_position_keys(occupied_pose_data)
    candidate_positions = torch.cartesian_prod(
        torch.arange(0, settings.camera.pose_l, dtype=torch.long),
        torch.arange(0, settings.camera.pose_w, dtype=torch.long),
        torch.arange(0, settings.camera.pose_h, dtype=torch.long),
        torch.arange(0, settings.camera.pose_n_elev, dtype=torch.long),
        torch.arange(0, settings.camera.pose_n_azim, dtype=torch.long),
    )

    candidates = []
    for pose_idx in candidate_positions:
        key = _pose_idx_key(pose_idx)
        if key in existing_keys:
            continue
        if _pose_idx_key(pose_idx)[:3] in occupied_position_keys:
            continue
        candidates.append(pose_idx)

    needed_count = requested_count - len(base_starts)
    if len(candidates) < needed_count:
        print(f"Warning: requested {requested_count} trajectories for {scene_name}, "
              f"but only {len(base_starts) + len(candidates)} non-duplicate start poses are available.")
        needed_count = len(candidates)

    if needed_count <= 0:
        return base_starts

    seed_bytes = hashlib.sha1(scene_name.encode('utf-8')).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder='big', signed=False) % (2 ** 32)
    rng = np.random.default_rng(seed)
    selected_indices = rng.permutation(len(candidates))[:needed_count]
    sampled_starts = torch.stack([candidates[int(i)] for i in selected_indices]).long().to(device)
    all_starts = torch.cat([base_starts, sampled_starts], dim=0)

    print(f"Generated {len(sampled_starts)} additional start positions for {scene_name}; "
          f"using {len(all_starts)} trajectories total.")
    return all_starts


def _save_hda_roi_export_frame(transmission_camera, frame_idx, images, depth):
    export_context = getattr(transmission_camera, 'hda_roi_export_context', None)
    if not export_context:
        return

    from PIL import Image

    trajectory_dir = export_context['trajectory_dir']
    damage_export_enabled = bool(export_context.get('damage_enabled', False))
    clean_dir = os.path.join(trajectory_dir, 'clean_images')
    pose_dir = os.path.join(trajectory_dir, 'poses')
    depth_dir = os.path.join(trajectory_dir, 'depth')
    valid_mask_dir = os.path.join(trajectory_dir, 'valid_masks')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(valid_mask_dir, exist_ok=True)

    frame_name = f"{frame_idx:06d}.png"
    image_np = images[0].detach().cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = image_np * 255.0
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    Image.fromarray(image_np, mode='RGB').save(os.path.join(clean_dir, frame_name))

    depth_np = depth[0, ..., 0].detach().cpu().numpy()
    valid = np.isfinite(depth_np) & (depth_np > 0)
    depth_u16 = np.zeros_like(depth_np, dtype=np.uint16)
    depth_scale = float(transmission_camera.zfar) / 65535.0
    if valid.any() and transmission_camera.zfar > 0:
        depth_u16[valid] = np.clip(
            depth_np[valid] / float(transmission_camera.zfar) * 65535.0,
            1,
            65535,
        ).astype(np.uint16)
    Image.fromarray(depth_u16, mode='I;16').save(os.path.join(depth_dir, frame_name))
    Image.fromarray(valid.astype(np.uint8) * 255, mode='L').save(
        os.path.join(valid_mask_dir, frame_name)
    )

    fov_camera = transmission_camera.fov_camera
    projection = fov_camera.get_full_projection_transform().get_matrix()
    view = fov_camera.get_world_to_view_transform().get_matrix()
    pose_metadata = {
        'scene_id': export_context.get('scene_id'),
        'damage_enabled': damage_export_enabled,
        'damage_config_id': export_context.get('damage_config_id') if damage_export_enabled else None,
        'trajectory_id': export_context.get('trajectory_id'),
        'frame_id': int(frame_idx),
        'image_name': frame_name,
        'rgb_image': os.path.join('clean_images', frame_name),
        'depth_image': os.path.join('depth', frame_name),
        'valid_mask': os.path.join('valid_masks', frame_name),
        'width': int(transmission_camera.image_width),
        'height': int(transmission_camera.image_height),
        'zfar': float(transmission_camera.zfar),
        'R': _to_jsonable(fov_camera.R),
        'T': _to_jsonable(fov_camera.T),
        'camera_center': _to_jsonable(fov_camera.get_camera_center()),
        'projection_matrix': _to_jsonable(projection),
        'world_to_view_matrix': _to_jsonable(view),
        'camera_model': 'pytorch3d_fov_perspective',
        'coordinate_system': 'MAGICIAN/PyTorch3D',
        'depth_encoding': {
            'format': 'uint16_png',
            'scale': depth_scale,
            'unit': 'scene_units',
            'invalid_value': 0,
            'depth_value': 'uint16_value * scale',
        },
        'valid_mask_encoding': {
            'format': 'uint8_png',
            'valid_value': 255,
            'invalid_value': 0,
        },
    }
    if hasattr(fov_camera, 'K') and fov_camera.K is not None:
        pose_metadata['K'] = _to_jsonable(fov_camera.K)

    pose_name = f"{frame_idx:06d}.json"
    pose_path = os.path.join(pose_dir, pose_name)
    with open(pose_path, 'w') as f:
        json.dump(pose_metadata, f, indent=2)

    frame_row = {
        'scene_id': export_context.get('scene_id'),
        'trajectory_id': export_context.get('trajectory_id'),
        'frame_id': int(frame_idx),
        'image_name': frame_name,
        'clean_image': os.path.join('clean_images', frame_name),
        'depth': os.path.join('depth', frame_name),
        'valid_mask': os.path.join('valid_masks', frame_name),
        'camera_pose': os.path.join('poses', pose_name),
        'width': int(transmission_camera.image_width),
        'height': int(transmission_camera.image_height),
    }
    if damage_export_enabled:
        frame_row['damage_config_id'] = export_context.get('damage_config_id')

    manifest_path = os.path.join(trajectory_dir, 'frame_manifest.jsonl')
    existing_rows = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                existing_rows = [json.loads(line) for line in f if line.strip()]
        except (OSError, json.JSONDecodeError):
            existing_rows = []
    existing_rows = [row for row in existing_rows if int(row.get('frame_id', -1)) != int(frame_idx)]
    existing_rows.append(frame_row)
    existing_rows.sort(key=lambda row: int(row.get('frame_id', -1)))
    with open(manifest_path, 'w') as f:
        for row in existing_rows:
            f.write(json.dumps(row) + '\n')

    summary = {
        'scene_id': export_context.get('scene_id'),
        'trajectory_id': export_context.get('trajectory_id'),
        'damage_enabled': damage_export_enabled,
        'num_frames': len(existing_rows),
        'base_observation_fields': ['clean_image', 'depth', 'valid_mask', 'camera_pose'],
    }
    if damage_export_enabled:
        summary['damage_config_id'] = export_context.get('damage_config_id')
    with open(os.path.join(trajectory_dir, 'trajectory_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def _hda_roi_trajectory_export_complete(trajectory_dir, save_depth=True, save_valid_mask=True):
    manifest_path = os.path.join(trajectory_dir, 'frame_manifest.jsonl')
    summary_path = os.path.join(trajectory_dir, 'trajectory_summary.json')
    if not os.path.exists(manifest_path) or not os.path.exists(summary_path):
        return False
    try:
        with open(manifest_path, 'r') as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if not rows:
        return False
    for row in rows:
        image_name = row.get('image_name')
        frame_id = row.get('frame_id')
        if not image_name or frame_id is None:
            return False
        if not os.path.exists(os.path.join(trajectory_dir, 'clean_images', image_name)):
            return False
        if not os.path.exists(os.path.join(trajectory_dir, 'poses', f"{int(frame_id):06d}.json")):
            return False
        if save_depth and not os.path.exists(os.path.join(trajectory_dir, 'depth', image_name)):
            return False
        if save_valid_mask and not os.path.exists(os.path.join(trajectory_dir, 'valid_masks', image_name)):
            return False
    return True


def _load_hda_roi_scene_damage_regions(export_root, scene_name, damage_config_id):
    config_path = os.path.join(export_root, 'scenes', scene_name, damage_config_id, 'damage_config.json')
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    regions = config.get('regions') or config.get('damage_regions')
    if not regions:
        return None
    damage_regions = dict(config)
    damage_regions['regions'] = regions
    return damage_regions


def capture_transmitted_highres_frame(mesh, planning_camera, transmission_camera):
    """
    Simulate a split UAV/server pipeline: the UAV captures and transmits a high-res
    frame, then the server downsamples that received frame for low-res planning.
    """
    highres_frame_idx = transmission_camera.n_frames_captured
    images, depth = transmission_camera.capture_image(mesh)
    _save_hda_roi_export_frame(transmission_camera, highres_frame_idx, images, depth)

    planning_images, planning_depth, planning_mask = _resize_frame_for_planning(
        images=images,
        depth=depth,
        planning_height=planning_camera.image_height,
        planning_width=planning_camera.image_width,
    )

    frame_idx = planning_camera.n_frames_captured
    frame_dict = {
        'rgb': planning_images,
        'zbuf': planning_depth,
        'mask': planning_mask,
        'R': planning_camera.fov_camera.R,
        'T': planning_camera.fov_camera.T,
        'zfar': planning_camera.zfar,
        'source_highres_frame_idx': highres_frame_idx,
        'source_highres_resolution': (
            transmission_camera.image_height,
            transmission_camera.image_width,
        ),
    }
    frame_save_path = os.path.join(planning_camera.save_dir_path, f"{frame_idx}.pt")
    torch.save(frame_dict, frame_save_path)
    planning_camera.n_frames_captured += 1


class SimpleGaussianModel:
    def __init__(self, means, opacities, scales, rotations, colors, device):
        """
        Args:
            means: (N, 3) locations
            opacities: (N, 1) opacity[0, 1]
            scales: (N, 3) 
            rotations: (N, 4) 
            colors: (N, 3) 
        """
        self.device = device
        self._xyz = means.to(device)
        self._opacity = self.inverse_sigmoid(opacities.to(device))  # logit
        self._scaling = torch.log(scales.to(device))  # log
        self._rotation = rotations.to(device)
        self._colors_precomp = colors.to(device)  

        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.max_radii2D = torch.zeros(means.shape[0], device=device)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-6):
        """ logit: logit(x) = log(x / (1-x))"""
        x = torch.clamp(x, eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # none
        return torch.zeros(self._xyz.shape[0], 1, 3, device=self.device)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_opacity_with_3D_filter(self):
        return self.get_opacity

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        return self.get_scaling, self.get_opacity

    @property
    def get_colors_precomp(self):
        return self._colors_precomp


def render_gaussian_depth(gaussian_means, gaussian_opacities, gaussian_scales,
                          gaussian_rotations, gaussian_colors, gs_camera, device,
                          bg_color=None, kernel_size=0.1):
    """

    Args:
        gaussian_means: (N, 3) 
        gaussian_opacities: (N, 1)
        gaussian_scales: (N, 3) 
        gaussian_rotations: (N, 4) 
        gaussian_colors: (N, 3) 
        gs_camera: GSCamera 
        device: torch device
        bg_color: 
        kernel_size: Mip-Splatting kernel size

    Returns:
        rendered_depth: (1, H, W) median depth
        rendered_image: (3, H, W) RGB image
    """
    if bg_color is None:
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    gaussians = SimpleGaussianModel(
        means=gaussian_means,
        opacities=gaussian_opacities,
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        colors=gaussian_colors,
        device=device
    )
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(gs_camera.image_height),
        image_width=int(gs_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=gs_camera.world_view_transform,
        projmatrix=gs_camera.full_proj_transform,
        sh_degree=0,
        campos=gs_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=True,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=device)
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    rotations = gaussians.get_rotation
    colors_precomp = gaussians.get_colors_precomp

    with torch.no_grad():
        rendered_image, radii, _, _, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

    return rendered_median_depth, rendered_image


def update_gaussian_colors_from_novelty(novelty_values):
    """
    use novelty_values update Gaussian colors

    Args:
        novelty_values: 
            0 = unknown → white [1,1,1]
            1 = known → black [0,0,0]
    """
    inverted_values = 1.0 - novelty_values
    colors = inverted_values.unsqueeze(1).repeat(1, 3)
    return colors


def update_gaussian_colors_for_damage_planning(novelty_values, remaining_damage_weight):
    """
    Pack general novelty and remaining damage value into one RGB render.
    Channel 0 keeps the original unseen signal; channel 1 is used for damage gain.
    """
    general_signal = 1.0 - novelty_values
    return torch.stack((general_signal, remaining_damage_weight, remaining_damage_weight), dim=1)


def _get_damage_config(params):
    return {
        'damage_aware_planning': bool(getattr(params, 'damage_aware_planning', False)),
        'damage_alpha': float(getattr(params, 'damage_alpha', 1.0)),
        'damage_beta': float(getattr(params, 'damage_beta', 5.0)),
        'damage_lambda_comm': float(getattr(params, 'damage_lambda_comm', 0.0)),
        'damage_gain_normalization': getattr(params, 'damage_gain_normalization', 'none'),
        'damage_gain_scale': float(getattr(params, 'damage_gain_scale', 1.0)),
        'damage_mode': getattr(params, 'damage_mode', 'random_3d_spheres'),
        'damage_seed': int(getattr(params, 'damage_seed', 1)),
        'damage_seed_scope': getattr(params, 'damage_seed_scope', 'scene'),
        'damage_num_regions': int(getattr(params, 'damage_num_regions', 5)),
        'damage_radius_ratio': float(getattr(params, 'damage_radius_ratio', 0.08)),
        'damage_soft_sigma_ratio': float(getattr(params, 'damage_soft_sigma_ratio', 0.04)),
        'damage_target_observations': int(getattr(params, 'damage_target_observations', 1)),
        'save_damage_debug': bool(getattr(params, 'save_damage_debug', True)),
        'save_damage_metrics': bool(getattr(params, 'save_damage_metrics', True)),
    }


def compute_damage_gain_score_scale(candidates, damage_config):
    mode = damage_config.get('damage_gain_normalization', 'none')
    configured_scale = float(damage_config.get('damage_gain_scale', 1.0))
    if mode in (None, 'none'):
        return configured_scale
    if mode == 'match_general_mean':
        general_values = np.asarray(
            [max(0.0, float(candidate['coverage_gain'])) for candidate in candidates],
            dtype=np.float64,
        )
        damage_values = np.asarray(
            [max(0.0, float(candidate['damage_gain'])) for candidate in candidates],
            dtype=np.float64,
        )
        positive_damage = damage_values[damage_values > 1e-8]
        if general_values.size == 0 or positive_damage.size == 0:
            return 0.0
        raw_scale = float(np.mean(general_values) / max(float(np.mean(positive_damage)), 1e-8))
        if configured_scale > 0:
            return min(raw_scale, configured_scale)
        return raw_scale
    raise ValueError(
        f"Unsupported damage_gain_normalization={mode}. Use 'none' or 'match_general_mean'."
    )


def apply_damage_gain_scoring(candidates, damage_config):
    scale = compute_damage_gain_score_scale(candidates, damage_config)
    for candidate in candidates:
        damage_gain_for_score = float(candidate['damage_gain']) * scale
        combined_gain = (
            damage_config['damage_alpha'] * float(candidate['coverage_gain'])
            + damage_config['damage_beta'] * damage_gain_for_score
            - damage_config['damage_lambda_comm'] * float(candidate.get('estimated_semantic_bits', 0.0))
        )
        candidate['damage_gain_score'] = damage_gain_for_score
        candidate['damage_gain_score_scale'] = scale
        candidate['combined_gain'] = combined_gain
        candidate['total_combined_gain'] = candidate['parent_total_combined_gain'] + combined_gain
        candidate['combined_gain_history'][-1] = combined_gain
        candidate['damage_gain_score_history'][-1] = damage_gain_for_score
    return scale


def _tensor_to_float_list(values):
    return torch.as_tensor(values).detach().cpu().float().tolist()


def _stable_damage_seed(base_seed, scene_name, trajectory_id, seed_scope='scene'):
    if seed_scope == 'scene':
        key = f"{base_seed}:{scene_name}".encode("utf-8")
    elif seed_scope == 'trajectory':
        key = f"{base_seed}:{scene_name}:{trajectory_id}".encode("utf-8")
    else:
        raise ValueError(
            f"Unsupported damage_seed_scope={seed_scope}. Use 'scene' or 'trajectory'."
        )
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def create_damage_regions(settings, damage_config, scene_name, trajectory_id, center_points,
                          center_source='gaussian_means'):
    if damage_config['damage_mode'] != 'random_3d_spheres':
        raise ValueError(
            f"Unsupported damage_mode={damage_config['damage_mode']}. "
            "Only random_3d_spheres is supported."
        )
    if center_points is None or center_points.shape[0] == 0:
        raise ValueError("Cannot create damage regions without center candidate points.")

    bbox_min = np.asarray(_tensor_to_float_list(settings.scene.x_min), dtype=np.float32)
    bbox_max = np.asarray(_tensor_to_float_list(settings.scene.x_max), dtype=np.float32)
    bbox_size = bbox_max - bbox_min
    scene_scale = float(np.linalg.norm(bbox_size))
    if scene_scale <= 0:
        scene_scale = float(np.max(bbox_size))

    radius = damage_config['damage_radius_ratio'] * scene_scale
    sigma = damage_config['damage_soft_sigma_ratio'] * scene_scale
    seed_scope = damage_config.get('damage_seed_scope', 'scene')
    rng_seed = _stable_damage_seed(
        damage_config['damage_seed'], scene_name, trajectory_id, seed_scope
    )
    rng = np.random.default_rng(rng_seed)
    candidate_points = center_points.detach().cpu().float().numpy()
    replace = candidate_points.shape[0] < damage_config['damage_num_regions']
    center_indices = rng.choice(
        candidate_points.shape[0],
        size=damage_config['damage_num_regions'],
        replace=replace,
    )
    centers = candidate_points[center_indices]

    return {
        'scene': scene_name,
        'trajectory_id': int(trajectory_id),
        'seed': int(damage_config['damage_seed']),
        'seed_scope': seed_scope,
        'resolved_seed': int(rng_seed),
        'center_source': center_source,
        'scene_bbox_min': bbox_min.tolist(),
        'scene_bbox_max': bbox_max.tolist(),
        'scene_scale': scene_scale,
        'num_center_candidates': int(candidate_points.shape[0]),
        'regions': [
            {
                'center': centers[i].astype(float).tolist(),
                'source_point_index': int(center_indices[i]),
                'radius': float(radius),
                'sigma': float(sigma),
            }
            for i in range(damage_config['damage_num_regions'])
        ],
    }


def compute_damage_weights(gaussian_means, damage_regions):
    if gaussian_means.shape[0] == 0 or len(damage_regions['regions']) == 0:
        return torch.zeros(gaussian_means.shape[0], device=gaussian_means.device)

    centers = torch.tensor(
        [region['center'] for region in damage_regions['regions']],
        dtype=gaussian_means.dtype,
        device=gaussian_means.device,
    )
    sigma = max(float(damage_regions['regions'][0]['sigma']), 1e-8)
    squared_dist = torch.cdist(gaussian_means, centers).pow(2)
    min_squared_dist = squared_dist.min(dim=1).values
    damage_weight = torch.exp(-min_squared_dist / (2.0 * sigma * sigma))
    return damage_weight.clamp(0.0, 1.0)


def damage_weight_stats(damage_weights):
    if damage_weights is None or damage_weights.numel() == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'sum': 0.0, 'nonzero_ratio': 0.0}
    return {
        'min': float(damage_weights.min().item()),
        'max': float(damage_weights.max().item()),
        'mean': float(damage_weights.mean().item()),
        'sum': float(damage_weights.sum().item()),
        'nonzero_ratio': float((damage_weights > 1e-6).float().mean().item()),
    }


def compute_damage_coverage(damage_weights, damage_observed_counts, target_observations):
    if damage_weights is None or damage_weights.numel() == 0:
        return 0.0
    total_damage_weight = damage_weights.sum().item()
    if total_damage_weight <= 1e-8:
        return 0.0
    covered_mask = damage_observed_counts >= max(1, target_observations)
    covered_weight = damage_weights[covered_mask].sum().item()
    return float(covered_weight / total_damage_weight)


def compute_damage_coverage_auc(curve):
    if len(curve) == 0:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    return float(np.trapz(np.asarray(curve, dtype=np.float64), dx=1.0) / (len(curve) - 1))


def write_damage_regions_json(path, damage_regions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(damage_regions, f, indent=2)


def write_damage_metrics_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'step',
        'general_coverage',
        'damage_coverage',
        'general_gain',
        'damage_gain',
        'damage_gain_score',
        'combined_gain',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})


def write_damage_points_ply(path, points, damage_weights, max_points=100000):
    selected = damage_weights > 0.5
    if selected.sum().item() == 0:
        selected = damage_weights > 1e-6
    selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
    if selected_indices.numel() > max_points:
        sample = torch.randperm(selected_indices.numel(), device=points.device)[:max_points]
        selected_indices = selected_indices[sample]

    selected_points = points[selected_indices].detach().cpu().float().numpy()
    selected_weights = damage_weights[selected_indices].detach().cpu().float().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {selected_points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float damage_weight\n")
        f.write("end_header\n")
        for point, weight in zip(selected_points, selected_weights):
            red = 255
            green = int(max(0, min(80, 80 * (1.0 - weight))))
            blue = int(max(0, min(80, 80 * (1.0 - weight))))
            f.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{red} {green} {blue} {float(weight)}\n"
            )


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(val) for val in value]
    if isinstance(value, tuple):
        return [_to_jsonable(val) for val in value]
    return value

# ==================== End RaDe-GS Integration ====================

def load_current_frame_perfect_depth(camera, device):
    current_frame_nb = camera.n_frames_captured - 1
    frame_path = os.path.join(camera.save_dir_path, str(current_frame_nb) + '.pt')

    
    frame_dict = torch.load(frame_path, map_location=device)
    
    return {
        'rgb': frame_dict['rgb'],           # (1, H, W, 3)
        'zbuf': frame_dict['zbuf'],         # (1, H, W, 1) 
        'mask': frame_dict['mask'],         # (1, H, W, 1)
        'R': frame_dict['R'],               # (1, 3, 3)
        'T': frame_dict['T'],               # (1, 3)
        'zfar': camera.zfar
    }

def apply_perfect_depth_simple(frame_data, device, use_error_mask=True):
    images = frame_data['rgb']
    zbuf = frame_data['zbuf'] 
    mask = frame_data['mask'].bool()
    R = frame_data['R']
    T = frame_data['T']
    
    # GT zbuf
    depth = torch.clamp(zbuf, min=0.5, max=750.0) 
    
    if use_error_mask:
        error_mask = mask
    else:
        error_mask = torch.ones_like(mask)
    
    return depth, mask, error_mask, R, T


def _valid_depth_mask(depth, mask):
    return mask.bool() & torch.isfinite(depth) & (depth > 0)


def _degrade_depth_block(valid, r, seed, block_size):
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    keep_mask = torch.zeros_like(valid, dtype=torch.bool)
    n_valid_total = int(valid.sum().item())
    target_keep_total = int(round(float(r) * n_valid_total))
    if target_keep_total <= 0:
        return keep_mask

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    kept_total = 0
    batch_size = valid.shape[0]
    height = valid.shape[1]
    width = valid.shape[2]

    for b in range(batch_size):
        valid_hw = valid[b, ..., 0] if valid.ndim == 4 else valid[b]
        blocks = []
        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                count = int(valid_hw[y0:y1, x0:x1].sum().item())
                if count > 0:
                    blocks.append((y0, y1, x0, x1, count))

        if not blocks:
            continue

        order = torch.randperm(len(blocks), generator=generator, device='cpu').tolist()
        for block_i in order:
            if kept_total >= target_keep_total:
                break
            y0, y1, x0, x1, count = blocks[block_i]
            current_gap = target_keep_total - kept_total
            overshoot = kept_total + count - target_keep_total
            if count <= current_gap or overshoot < current_gap:
                if valid.ndim == 4:
                    keep_mask[b, y0:y1, x0:x1, :] = valid[b, y0:y1, x0:x1, :]
                else:
                    keep_mask[b, y0:y1, x0:x1] = valid[b, y0:y1, x0:x1]
                kept_total += count

    return keep_mask


def _degrade_depth_pixel_dropout(valid, r, seed):
    """Uniformly drop individual valid pixels to retain r fraction."""
    n_valid = int(valid.sum().item())
    target_keep = int(round(float(r) * n_valid))
    keep_mask = torch.zeros_like(valid, dtype=torch.bool)
    if target_keep <= 0:
        return keep_mask
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    valid_flat = valid.reshape(-1)
    valid_indices = valid_flat.nonzero(as_tuple=False).squeeze(1)
    perm = torch.randperm(len(valid_indices), generator=generator)
    keep_indices = valid_indices[perm[:target_keep]]
    keep_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
    keep_flat[keep_indices] = True
    return keep_flat.view(valid.shape)


def _degrade_depth_downsample_upsample(depth, mask, scale):
    """Bilinear-downsample to scale×H, scale×W then nearest-upsample back to original size.

    scale is the linear spatial scale factor (0 < scale <= 1.0).
    Depth uses area-mode down, nearest-mode up. Mask uses nearest throughout.
    """
    import torch.nn.functional as _F
    scale = float(scale)
    if scale >= 1.0:
        return depth.clone(), mask.bool()
    if scale <= 0.0:
        return depth.clone(), torch.zeros_like(mask, dtype=torch.bool)

    ndim = depth.ndim
    if ndim == 4:
        depth_4d = depth.permute(0, 3, 1, 2).float()
        mask_4d = mask.float().permute(0, 3, 1, 2) if mask.ndim == 4 else mask.float().unsqueeze(1)
    else:
        depth_4d = depth.float().unsqueeze(1)
        mask_4d = mask.float().unsqueeze(1)

    _B, _C, H, W = depth_4d.shape
    H_low = max(1, int(H * scale))
    W_low = max(1, int(W * scale))

    depth_low = _F.interpolate(depth_4d, size=(H_low, W_low), mode='area')
    mask_low = _F.interpolate(mask_4d, size=(H_low, W_low), mode='nearest')
    depth_up = _F.interpolate(depth_low, size=(H, W), mode='nearest')
    mask_up = _F.interpolate(mask_low, size=(H, W), mode='nearest')

    # Prevent invalid-to-valid promotion: AND with original mask.
    mask_up = mask_up * mask_4d

    if ndim == 4:
        return depth_up.permute(0, 2, 3, 1), mask_up.permute(0, 2, 3, 1).bool()
    return depth_up.squeeze(1), mask_up.squeeze(1).bool()


def degrade_depth(depth, mask, r, mode="block", seed=0, block_size=16):
    r = float(r)
    if r < 0.0 or r > 1.0:
        raise ValueError(f"r must be in [0, 1], got {r}.")

    valid = _valid_depth_mask(depth, mask)
    n_valid = int(valid.sum().item())
    if n_valid == 0 or r >= 1.0:
        return depth, mask.bool()

    if mode == "block":
        keep_mask = _degrade_depth_block(valid, r=r, seed=seed, block_size=int(block_size))
        degraded_depth = depth.clone()
        degraded_mask = mask.bool() & keep_mask
        degraded_depth[mask.bool() & ~keep_mask] = 0.
        return degraded_depth, degraded_mask
    elif mode == "pixel_dropout":
        keep_mask = _degrade_depth_pixel_dropout(valid, r=r, seed=seed)
        degraded_depth = depth.clone()
        degraded_mask = mask.bool() & keep_mask
        degraded_depth[mask.bool() & ~keep_mask] = 0.
        return degraded_depth, degraded_mask
    elif mode == "downsample_upsample":
        # r is interpreted as the linear spatial scale factor.
        return _degrade_depth_downsample_upsample(depth, mask, scale=r)
    else:
        raise ValueError(
            f"Unsupported depth degradation mode: {mode!r}. "
            "Supported: 'block', 'pixel_dropout', 'downsample_upsample'."
        )

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def setup_test(params, model_path, device, verbose=True):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path)
    print("\nThe following scenes will be used to test the model:")
    for batch, elem in enumerate(test_dataloader):
        print(elem['scene_name'][0])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)


    trained_weights = torch.load(model_path, map_location=device, weights_only=False)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True)  # todo: replace by params.ddp
    depth_losses = np.array(trained_weights["depth_losses"])
    depth_losses_per_epoch = (depth_losses[::2] + depth_losses[1::2]) / 2
    # depth_losses_per_epoch = depth_losses
    print("\nModel name:", model_path)
    print("\nThe model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")
    print("It has been trained for", trained_weights["epoch"], "epochs.")
    print("The loss was:", depth_losses_per_epoch[-1], depth_losses_per_epoch[-1] * 3 / 4)
    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0, verbose=verbose)

    return test_dataloader, macarons, memory


def setup_test_scene(params,
                     mesh,
                     settings,
                     mirrored_scene,
                     device,
                     mirrored_axis=None,
                     surface_scene_feature_dim=1,
                     test_resolution=0.05,
                     covered_scene_feature_dim=1):
    """
    Setup the different scene objects used for prediction and performance evaluation.

    :param params:
    :param mesh:
    :param settings:
    :param device:
    :param is_master:
    :return:
    """

    # Initialize gt_scene: we use this scene to store gt surface points to evaluate the performance of the model.
    # This scene is not used for supervision during training, since the model is self-supervised from RGB data
    # captured in real-time.
    gt_scene = Scene(x_min=settings.scene.x_min,
                     x_max=settings.scene.x_max,
                     grid_l=settings.scene.grid_l,
                     grid_w=settings.scene.grid_w,
                     grid_h=settings.scene.grid_h,
                     cell_capacity=params.surface_cell_capacity,
                     cell_resolution=test_resolution * params.scene_scale_factor,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    covered_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=test_resolution * params.scene_scale_factor,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=covered_scene_feature_dim,
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface, gt_normals, gt_surface_colors = get_scene_gt_surface(gt_scene=gt_scene,
                                                         verts=mesh.verts_list()[0],
                                                         faces=mesh.faces_list()[0],
                                                         n_surface_points=params.n_gt_surface_points,
                                                         return_colors=True,
                                                         mesh=mesh)
    gt_scene.fill_cells(gt_surface, features=gt_surface_colors)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth model from RGB images
    surface_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=None,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=surface_scene_feature_dim,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points
    proxy_scene = Scene(x_min=settings.scene.x_min,
                        x_max=settings.scene.x_max,
                        grid_l=settings.scene.grid_l,
                        grid_w=settings.scene.grid_w,
                        grid_h=settings.scene.grid_h,
                        cell_capacity=params.proxy_cell_capacity,
                        cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points,
                        device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1,  # We use proxy points indices as features
                        mirrored_scene=mirrored_scene,
                        score_threshold=params.score_threshold,
                        mirrored_axis=mirrored_axis)
    proxy_scene.initialize_proxy_points()

    return gt_scene, covered_scene, surface_scene, proxy_scene


def setup_test_camera(params,
                      mesh, intersector, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None,
                      image_height=None,
                      image_width=None,
                      capture_initial=True):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    if image_height is None:
        image_height = params.image_height
    if image_width is None:
        image_width = params.image_width

    renderer = get_rgb_renderer(image_height=image_height,
                                image_width=image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=200000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test


    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=start_cam_idx)

    if capture_initial:
        camera.capture_image(mesh)

    return camera


def compute_magician_trajectory(params, macarons, camera, gt_scene, surface_scene,
                           proxy_scene, covered_scene, mesh, intersector, device, settings,
                           test_resolution=0.05, use_perfect_depth_map=False,
                           compute_collision=False,
                           transmission_camera=None, scene_name=None, trajectory_id=0,
                           preset_damage_regions=None):

    macarons.eval()

    # compute scene_scales
    scene_bbox_x = settings.scene.x_max[0] - settings.scene.x_min[0]
    scene_bbox_y = settings.scene.x_max[1] - settings.scene.x_min[1]
    scene_bbox_z = settings.scene.x_max[2] - settings.scene.x_min[2]
    scene_scale = (scene_bbox_x + scene_bbox_y + scene_bbox_z) / 3.0
    print(f"Scene scale computed: {scene_scale:.2f} (bbox: x={scene_bbox_x:.2f}, y={scene_bbox_y:.2f}, z={scene_bbox_z:.2f})")

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)
    full_pc_idx = torch.zeros(0, 1, device=device)
    coverage_evolution = []
    timing_records = []
    communication_records = []
    damage_metric_rows = []
    damage_coverage_curve = []
    damage_gain_accumulated = 0.0
    damage_gain_score_accumulated = 0.0
    combined_gain_accumulated = 0.0
    selected_general_gain_accumulated = 0.0
    last_processed_frame = 0
    pose_i = 0

    damage_config = _get_damage_config(params)
    damage_enabled = damage_config['damage_aware_planning']
    damage_regions = None
    latest_damage_weights = None
    latest_damage_weight_stats = None
    damage_debug_dir = os.path.join(os.path.dirname(camera.save_dir_path), 'damage_debug')
    damage_points_saved = False
    if damage_enabled:
        scene_name_for_seed = scene_name if scene_name is not None else "unknown_scene"
        print(
            "Damage-aware planning enabled: "
            f"regions={damage_config['damage_num_regions']}, "
            f"alpha={damage_config['damage_alpha']}, "
            f"beta={damage_config['damage_beta']}, "
            f"target_observations={damage_config['damage_target_observations']}, "
            f"seed_scope={damage_config['damage_seed_scope']}, "
            f"gain_norm={damage_config['damage_gain_normalization']}, "
            f"gain_scale={damage_config['damage_gain_scale']}"
        )
    else:
        print("Damage-aware planning disabled.")
    
    def load_frame(frame_idx):
        frame_path = os.path.join(camera.save_dir_path, str(frame_idx) + '.pt')
        frame_dict = torch.load(frame_path, map_location=device)
        return {
            'rgb': frame_dict['rgb'],
            'zbuf': frame_dict['zbuf'],
            'mask': frame_dict['mask'],
            'R': frame_dict['R'],
            'T': frame_dict['T'],
            'zfar': camera.zfar,
            'frame_idx': frame_idx,
        }

    def process_frame(frame_idx):
        current_frame = load_frame(frame_idx)
        depth, mask, error_mask, R, T = apply_perfect_depth_simple(current_frame, device)
        n_valid_depth_before_degrade = int(_valid_depth_mask(depth, mask).sum().item())
        depth, mask = degrade_depth(
            depth, mask,
            r=getattr(params, 'r_aux', 1.0),
            mode=getattr(params, 'degrade_mode', 'block'),
            seed=int(getattr(params, 'degrade_seed', 0)) + int(frame_idx),
            block_size=int(getattr(params, 'degrade_block_size', 16)),
        )
        n_valid_depth_after_degrade = int(_valid_depth_mask(depth, mask).sum().item())
        
        fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        X_cam = fov_camera.get_camera_center() 
        
        part_pc, part_pc_features = camera.compute_partial_point_cloud(
            depth=depth, mask=(mask * error_mask).bool(), images=current_frame['rgb'],
            fov_cameras=fov_camera,
            gathering_factor=params.gathering_factor * 2,
            fov_range=params.sensor_range
        )
        
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(
            proxy_scene.proxy_points, return_mask=True,
            fov_camera=fov_camera, fov_range=params.sensor_range
        )
        
        sgn_dists = None
        if fov_proxy_mask.any():
            sgn_dists = camera.get_signed_distance_to_depth_maps(
                pts=fov_proxy_points, depth_maps=depth,
                mask=mask, fov_camera=fov_camera
            )
        
        return {
            'part_pc': part_pc,
            'part_pc_features': part_pc_features,
            'fov_proxy_points': fov_proxy_points,
            'fov_proxy_mask': fov_proxy_mask,
            'sgn_dists': sgn_dists,
            'X_cam': X_cam,
            'current_frame': current_frame,
            'n_valid_depth_before_degrade': n_valid_depth_before_degrade,
            'n_valid_depth_after_degrade': n_valid_depth_after_degrade,
        }

    def update_scene_with_frame(frame_data, frame_idx):
        nonlocal full_pc, full_pc_colors, full_pc_idx

        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))
        full_pc_colors = torch.vstack((full_pc_colors, frame_data['part_pc_features']))
        part_pc_idx = torch.full((frame_data['part_pc'].shape[0], 1), frame_idx, device=device)
        full_pc_idx = torch.vstack((full_pc_idx, part_pc_idx))

        if frame_data['fov_proxy_mask'].any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data['fov_proxy_mask'])
            proxy_scene.fill_cells(frame_data['fov_proxy_points'],
                                   features=fov_proxy_indices.view(-1, 1))

            proxy_scene.update_proxy_view_states(
                camera, frame_data['fov_proxy_mask'],
                signed_distances=frame_data['sgn_dists'],
                distance_to_surface=None,
                X_cam=frame_data['X_cam']
            )

            proxy_scene.update_proxy_supervision_occ(
                frame_data['fov_proxy_mask'], frame_data['sgn_dists'],
                tol=params.carving_tolerance
            )
            proxy_scene.update_proxy_out_of_field(frame_data['fov_proxy_mask'])
            

    while pose_i <= params.n_poses_in_trajectory:
        iteration_start = _profile_now(device)
        timing_record = {
            'method': 'MAGICIAN',
            'pose_i': pose_i,
            'num_camera_history_before_planning': len(camera.X_cam_history),
            'beam_width': params.beam_width,
            'beam_steps': params.beam_steps,
        }
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)

        stage_start = _profile_now(device)
        new_frame_indices = list(range(last_processed_frame, camera.n_frames_captured))
        frame_data_list = [process_frame(frame_idx) for frame_idx in new_frame_indices]
        timing_record['observation_time_s'] = _profile_now(device) - stage_start
        timing_record['new_frames_processed'] = len(new_frame_indices)
        timing_record['processed_frame_indices'] = new_frame_indices
        n_valid_depth_before = [frame_data['n_valid_depth_before_degrade'] for frame_data in frame_data_list]
        n_valid_depth_after = [frame_data['n_valid_depth_after_degrade'] for frame_data in frame_data_list]
        timing_record['n_valid_depth_points_before_degrade_per_frame'] = n_valid_depth_before
        timing_record['n_valid_depth_points_per_frame'] = n_valid_depth_after
        timing_record['n_valid_depth_points_before_degrade'] = int(sum(n_valid_depth_before))
        timing_record['n_valid_depth_points'] = int(sum(n_valid_depth_after))
        timing_record['depth_retention_ratio'] = (
            timing_record['n_valid_depth_points']
            / max(1, timing_record['n_valid_depth_points_before_degrade'])
        )
        timing_record['r_aux'] = float(getattr(params, 'r_aux', 1.0))
        timing_record['degrade_mode'] = getattr(params, 'degrade_mode', 'block')
        timing_record['degrade_seed'] = int(getattr(params, 'degrade_seed', 0))
        timing_record['degrade_block_size'] = int(getattr(params, 'degrade_block_size', 16))

        # Update scene with every newly captured interpolation frame.
        stage_start = _profile_now(device)
        comm_totals = {
            'raw_pt_bytes': 0,
            'rgb_png_bytes': 0,
            'depth_uint16_bytes': 0,
            'mask_bitpacked_bytes': 0,
            'pose_intrinsics_bytes': 0,
            'lossless_sensor_est_bytes': 0,
        }
        for frame_idx, frame_data in zip(new_frame_indices, frame_data_list):
            update_scene_with_frame(frame_data, frame_idx)
            communication_camera = transmission_camera if transmission_camera is not None else camera
            frame_comm = _frame_communication_bytes(communication_camera, frame_idx)
            frame_comm.update({
                'pose_i': pose_i,
                'frame_idx': frame_idx,
                'method': 'MAGICIAN',
                'n_interpolation_steps': camera.n_interpolation_steps,
                'planning_image_height': camera.image_height,
                'planning_image_width': camera.image_width,
                'transmitted_image_height': communication_camera.image_height,
                'transmitted_image_width': communication_camera.image_width,
                'transmission_mode': 'highres_downsampled_for_planning'
                if transmission_camera is not None else 'planning_resolution',
            })
            communication_records.append(frame_comm)
            for key in comm_totals:
                comm_totals[key] += frame_comm[key]
        last_processed_frame = camera.n_frames_captured

        surface_scene.set_all_features_to_value(value=1.)
        timing_record['map_update_time_s'] = _profile_now(device) - stage_start
        timing_record['communication_raw_pt_bytes'] = int(comm_totals['raw_pt_bytes'])
        timing_record['communication_rgb_png_bytes'] = int(comm_totals['rgb_png_bytes'])
        timing_record['communication_depth_uint16_bytes'] = int(comm_totals['depth_uint16_bytes'])
        timing_record['communication_mask_bitpacked_bytes'] = int(comm_totals['mask_bitpacked_bytes'])
        timing_record['communication_pose_intrinsics_bytes'] = int(comm_totals['pose_intrinsics_bytes'])
        timing_record['communication_lossless_sensor_est_bytes'] = int(comm_totals['lossless_sensor_est_bytes'])

        # Compute coverage gain for evaulation
        stage_start = _profile_now(device)
        current_coverage = gt_scene.scene_coverage(
            covered_scene, surface_epsilon=2 * test_resolution * params.scene_scale_factor
        )
        if pose_i % 5 == 0:
            print("==========current coverage:", current_coverage)
        current_cov = current_coverage[0].item() if current_coverage[0] != 0. else 0.
        coverage_evolution.append(current_cov / settings.scene.visibility_ratio)
        timing_record['coverage_eval_time_s'] = _profile_now(device) - stage_start

        # Occupancy field prediction
        stage_start = _profile_now(device)
        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
                params, macarons.scone, camera, surface_scene, proxy_scene, device
            )
        timing_record['occupancy_inference_time_s'] = _profile_now(device) - stage_start
        # We only keep the points with occupancy value larger than 0.5
        filtered_X_world = X_world[occ_probs.squeeze() > 0.5]
        n_points = filtered_X_world.shape[0]
        timing_record['num_imagined_gaussians'] = int(n_points)
        gaussian_means = filtered_X_world  # (N, 3) 
        occ_values = occ_probs[occ_probs.squeeze() > 0.5] 

        # Convert occupancy field to Imagined Gaussians
        gaussian_opacities = occ_values    # (N, 1) 
        gaussian_scales = torch.ones(n_points, 3, device=device) * (0.7154/2)  
        gaussian_rotations = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(n_points, 1) 
        novelty_values = torch.zeros(n_points, device=device)  # (N,)
        gaussian_colors = update_gaussian_colors_from_novelty(novelty_values)  # (N, 3)
        damage_weights = None
        damage_observed_counts = None
        if damage_enabled:
            if damage_regions is None:
                scene_name_for_seed = scene_name if scene_name is not None else "unknown_scene"
                if preset_damage_regions is not None:
                    damage_regions = preset_damage_regions
                    print(
                        "Reusing scene-level damage regions: "
                        f"regions={len(damage_regions['regions'])}, "
                        f"center_source={damage_regions.get('center_source', 'unknown')}"
                    )
                else:
                    center_points = gaussian_means
                    center_source = 'gaussian_means'
                    if center_points.shape[0] == 0 and full_pc.shape[0] > 0:
                        center_points = full_pc
                        center_source = 'observed_point_cloud'
                    if center_points.shape[0] == 0:
                        center_points = mesh.verts_list()[0]
                        center_source = 'mesh_vertices'
                    damage_regions = create_damage_regions(
                        settings=settings,
                        damage_config=damage_config,
                        scene_name=scene_name_for_seed,
                        trajectory_id=trajectory_id,
                        center_points=center_points,
                        center_source=center_source,
                    )
                    print(
                        "Damage regions sampled from "
                        f"{center_source}: candidates={damage_regions['num_center_candidates']}, "
                        f"regions={len(damage_regions['regions'])}"
                    )
                if damage_config['save_damage_debug']:
                    write_damage_regions_json(
                        os.path.join(damage_debug_dir, 'damage_regions.json'),
                        damage_regions,
                    )
            damage_weights = compute_damage_weights(gaussian_means, damage_regions)
            damage_observed_counts = torch.zeros(n_points, device=device, dtype=torch.float32)
            latest_damage_weights = damage_weights.detach().clone()
            latest_damage_weight_stats = damage_weight_stats(damage_weights)
            timing_record['damage_weight_stats'] = latest_damage_weight_stats
            print(
                "Damage weights: "
                f"min={latest_damage_weight_stats['min']:.4f}, "
                f"max={latest_damage_weight_stats['max']:.4f}, "
                f"mean={latest_damage_weight_stats['mean']:.4f}, "
                f"nonzero={latest_damage_weight_stats['nonzero_ratio']:.4f}"
            )
            if damage_config['save_damage_debug'] and not damage_points_saved:
                write_damage_points_ply(
                    os.path.join(damage_debug_dir, 'damage_points.ply'),
                    gaussian_means,
                    damage_weights,
                )
                damage_points_saved = True

        if pose_i == 0:
            sample_X_cam = camera.X_cam_history[0].view(1, 3)
            sample_V_cam = camera.V_cam_history[0].view(1, 2)
            R_sample, T_sample = get_camera_RT(sample_X_cam, sample_V_cam)
            sample_camera = FoVPerspectiveCameras(R=R_sample, T=T_sample, zfar=camera.zfar, device=device)
            K_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)

        # 1. initialize all novelty_values to 0
        novelty_values = torch.zeros(n_points, device=device)

        # 2. revisit all previous cameras
        stage_start = _profile_now(device)
        history_length = len(camera.X_cam_history)

        for cam_idx in range(history_length):
            current_X_cam = camera.X_cam_history[cam_idx]
            current_V_cam = camera.V_cam_history[cam_idx]
            X_cam = current_X_cam.view(1, 3)
            V_cam = current_V_cam.view(1, 2)
            R_cam, T_cam = get_camera_RT(X_cam, V_cam)
            current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)
            current_fov_camera.K = K_matrix  

            gs_cameras = convert_camera_from_pytorch3d_to_gs(
                current_fov_camera,
                height=camera.image_height,
                width=camera.image_width,
                device=device
            )
            gs_camera = gs_cameras[0]

            with torch.no_grad():
                rendered_depth, _ = render_gaussian_depth(
                    gaussian_means=gaussian_means,
                    gaussian_opacities=gaussian_opacities,
                    gaussian_scales=gaussian_scales,
                    gaussian_rotations=gaussian_rotations,
                    gaussian_colors=gaussian_colors,
                    gs_camera=gs_camera,
                    device=device,
                    bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                    kernel_size=0.01
                )
                current_depth_map = rendered_depth[0]

                current_visible_mask = camera.check_point_visibility_from_depth(
                    filtered_X_world, current_fov_camera, current_depth_map, depth_tolerance=1.0
                )
                # update the novelty along the visited cameras
                novelty_values[current_visible_mask] = 1.0
                if damage_enabled:
                    damage_observed_counts[current_visible_mask] += 1.0

        print(f"historical: {novelty_values.sum().item()}/{n_points}")
        current_damage_coverage = None
        if damage_enabled:
            target_observations = max(1, damage_config['damage_target_observations'])
            damage_observed_counts.clamp_(max=float(target_observations))
            current_damage_coverage = compute_damage_coverage(
                damage_weights,
                damage_observed_counts,
                target_observations,
            )
            print(f"current damage coverage: {current_damage_coverage:.4f}")
        timing_record['history_visibility_update_time_s'] = _profile_now(device) - stage_start

        # 3. Beam Search 
        stage_start = _profile_now(device)
        remaining_steps = params.n_poses_in_trajectory + 1 - history_length
        print(f"Beam search remain: {remaining_steps} steps")

        # initialize beam search
        initial_pose_idx = camera.cam_idx
        beams = [{
            'trajectory': [],
            'novelty_values': novelty_values.clone(),
            'damage_observed_counts': damage_observed_counts.clone() if damage_enabled else None,
            'score': novelty_values.sum().item(),
            'total_coverage_gain': 0.0,  
            'total_damage_gain': 0.0,
            'total_combined_gain': 0.0,
            'gain_history': [],
            'damage_gain_history': [],
            'damage_gain_score_history': [],
            'combined_gain_history': [],
            'damage_coverage_history': [],
            'current_pose_idx': initial_pose_idx
        }]

        # settings for beam search
        beam_width = params.beam_width
        total_candidate_evaluations = 0
        total_valid_motion_candidates = 0
        for bs_i in range(params.beam_steps):
            print(f"Beam search step {bs_i + 1}/{params.beam_steps}")

            all_candidates = []

            # extend to every beams
            for beam in beams:
                neighbor_indices = camera.get_neighboring_poses(pose_idx=beam['current_pose_idx'])
                valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

                rendering_candidate = []
                idx_candidate = []

                current_pose, _ = camera.get_pose_from_idx(beam['current_pose_idx'])
                X_current, _, _ = camera.get_camera_parameters_from_pose(current_pose)
                current_loc = X_current[0].cpu().numpy()

                for row in valid_neighbors:
                    neighbor_pose, _ = camera.get_pose_from_idx(row)
                    X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    target_loc = X_neighbor[0].cpu().numpy()

                    if compute_collision:
                        if bs_i == 0:
                            if line_segment_mesh_intersection(current_loc, target_loc, intersector):
                                continue
                        else:
                            # Use occupancy points to check for future collisions.
                            if line_segment_intersects_point_cloud_region(filtered_X_world, X_current[0], X_neighbor[0]):
                                continue

                    rendering_candidate.append(fov_neighbor)
                    idx_candidate.append(row)

                if len(rendering_candidate) == 0:
                    continue
                total_valid_motion_candidates += len(rendering_candidate)

                # rendering for every pose
                for j, pose_idx in enumerate(idx_candidate):
                    total_candidate_evaluations += 1
                    fov_camera = rendering_candidate[j]
                    fov_camera.K = K_matrix

                    gs_cameras = convert_camera_from_pytorch3d_to_gs(
                        fov_camera,
                        height=camera.image_height,
                        width=camera.image_width,
                        device=device
                    )
                    gs_camera = gs_cameras[0]

                    # update colors
                    current_novelty= beam['novelty_values']
                    current_damage_counts = beam['damage_observed_counts'] if damage_enabled else None
                    if damage_enabled:
                        target_observations = max(1, damage_config['damage_target_observations'])
                        remaining_damage_weight = damage_weights * torch.clamp(
                            (target_observations - current_damage_counts) / target_observations,
                            min=0.0,
                            max=1.0,
                        )
                        gaussian_colors = update_gaussian_colors_for_damage_planning(
                            current_novelty,
                            remaining_damage_weight,
                        )
                    else:
                        gaussian_colors = update_gaussian_colors_from_novelty(current_novelty)

                    with torch.no_grad():
                        rendered_depth, rendered_image = render_gaussian_depth(
                            gaussian_means=gaussian_means,
                            gaussian_opacities=gaussian_opacities,
                            gaussian_scales=gaussian_scales,
                            gaussian_rotations=gaussian_rotations,
                            gaussian_colors=gaussian_colors,
                            gs_camera=gs_camera,
                            device=device,
                            bg_color=torch.tensor(
                                [1.0, 0.0, 0.0] if damage_enabled else [1.0, 1.0, 1.0],
                                device=device
                            ),
                            kernel_size=0.01
                        )
                        depth_map = rendered_depth[0]

                        # compute visible mask
                        visible_mask = camera.check_point_visibility_from_depth(
                            filtered_X_world, fov_camera, depth_map, depth_tolerance=1.0
                        )

                        # white pixels: unseen points（novelty_values=0）
                        valid_depth_mask = depth_map > 0
                        rgb_image = rendered_image  # shape: [3, H, W]
                        if damage_enabled:
                            general_map = rgb_image[0]
                            damage_map = rgb_image[1]
                        else:
                            grayscale = rgb_image.mean(dim=0)  # [H, W]

               
                        depth_threshold = scene_scale / 2.0  

                        if valid_depth_mask.any():
                            nb_observed_pts_per_pixel = (depth_map / depth_threshold) ** 2
                            depth_weight = nb_observed_pts_per_pixel.clamp_max(1.0)

                            # compute coverage gain by using novelty map and depth weights map
                            if damage_enabled:
                                coverage_gain = (
                                    general_map * depth_weight * valid_depth_mask.float()
                                ).sum().item()
                                damage_gain = (
                                    damage_map * depth_weight * valid_depth_mask.float()
                                ).sum().item()
                            else:
                                coverage_gain = (grayscale * depth_weight * valid_depth_mask.float()).sum().item()
                                damage_gain = 0.0
                        else:
                            coverage_gain = 0.0
                            damage_gain = 0.0

                        estimated_semantic_bits = 0.0
                        if damage_enabled:
                            damage_gain_score = damage_gain
                            combined_gain = 0.0
                        else:
                            damage_gain_score = 0.0
                            combined_gain = coverage_gain

                        new_novelty = current_novelty.clone()
                        new_novelty[visible_mask] = 1.0
                        if damage_enabled:
                            new_damage_counts = current_damage_counts.clone()
                            new_damage_counts[visible_mask] += 1.0
                            new_damage_counts.clamp_(max=float(target_observations))
                        else:
                            new_damage_counts = None

                        new_total_coverage_gain = beam['total_coverage_gain'] + coverage_gain
                        new_total_damage_gain = beam['total_damage_gain'] + damage_gain
                        new_total_combined_gain = beam['total_combined_gain'] + combined_gain
                        if damage_enabled:
                            new_damage_coverage = compute_damage_coverage(
                                damage_weights,
                                new_damage_counts,
                                target_observations,
                            )
                        else:
                            new_damage_coverage = 0.0

                        all_candidates.append({
                            'trajectory': beam['trajectory'] + [pose_idx],
                            'novelty_values': new_novelty,
                            'damage_observed_counts': new_damage_counts,
                            'coverage_gain': coverage_gain,  # single step
                            'damage_gain': damage_gain,
                            'damage_gain_score': damage_gain_score,
                            'combined_gain': combined_gain,
                            'total_coverage_gain': new_total_coverage_gain, 
                            'total_damage_gain': new_total_damage_gain,
                            'total_combined_gain': new_total_combined_gain,
                            'parent_total_combined_gain': beam['total_combined_gain'],
                            'estimated_semantic_bits': estimated_semantic_bits,
                            'gain_history': beam['gain_history'] + [coverage_gain],
                            'damage_gain_history': beam['damage_gain_history'] + [damage_gain],
                            'damage_gain_score_history': beam['damage_gain_score_history'] + [damage_gain_score],
                            'combined_gain_history': beam['combined_gain_history'] + [combined_gain],
                            'damage_coverage_history': beam['damage_coverage_history'] + [new_damage_coverage],
                            'current_pose_idx': pose_idx
                        })

            if len(all_candidates) == 0:
                print("No valid candidates found!")
                break

            damage_gain_score_scale = 1.0
            if damage_enabled:
                damage_gain_score_scale = apply_damage_gain_scoring(all_candidates, damage_config)

            # coverage gains based on rgb imgs, optionally reweighted by damage gain
            all_candidates.sort(
                key=lambda x: x['total_combined_gain'] if damage_enabled else x['total_coverage_gain'],
                reverse=True
            )
            beams = all_candidates[:beam_width]
            # print(f"Step {bs_i + 1}: Best total_coverage_gain = {beams[0]['total_coverage_gain']:.2f}, Score = {beams[0]['score']}/{n_points}, Current step gain = {beams[0].get('coverage_gain', 0):.2f}")
            print(f"Top {min(beam_width, len(all_candidates))} beams selected from {len(all_candidates)} candidates")
            if damage_enabled and len(beams) > 0:
                top_general = np.mean([beam['coverage_gain'] for beam in beams])
                top_damage = np.mean([beam['damage_gain'] for beam in beams])
                top_damage_score = np.mean([beam['damage_gain_score'] for beam in beams])
                top_combined = np.mean([beam['combined_gain'] for beam in beams])
                print(
                    "Damage-aware beam gains: "
                    f"general_mean={top_general:.2f}, "
                    f"damage_raw_mean={top_damage:.2f}, "
                    f"damage_score_mean={top_damage_score:.2f}, "
                    f"damage_scale={damage_gain_score_scale:.2f}, "
                    f"combined_mean={top_combined:.2f}"
                )
        timing_record['beam_search_time_s'] = _profile_now(device) - stage_start
        timing_record['beam_candidate_evaluations'] = int(total_candidate_evaluations)
        timing_record['valid_motion_candidates'] = int(total_valid_motion_candidates)

        if len(beams) > 0 and len(beams[0]['trajectory']) > 0:
            best_beam = beams[0]
            best_trajectory = best_beam['trajectory']
        else:
            print("No valid trajectory found!")
            timing_records.append(timing_record)
            break

        # move one step
        next_idx = best_trajectory[0]
        print(f"move one step: pose_idx = {next_idx}")
        timing_record['selected_next_idx'] = next_idx.detach().cpu().numpy().tolist()
        selected_general_gain = best_beam['gain_history'][0] if len(best_beam['gain_history']) > 0 else 0.0
        selected_damage_gain = (
            best_beam['damage_gain_history'][0]
            if damage_enabled and len(best_beam['damage_gain_history']) > 0
            else 0.0
        )
        selected_damage_gain_score = (
            best_beam['damage_gain_score_history'][0]
            if damage_enabled and len(best_beam.get('damage_gain_score_history', [])) > 0
            else selected_damage_gain
        )
        selected_combined_gain = (
            best_beam['combined_gain_history'][0]
            if len(best_beam['combined_gain_history']) > 0
            else selected_general_gain
        )
        selected_general_gain_accumulated += selected_general_gain
        damage_gain_accumulated += selected_damage_gain
        damage_gain_score_accumulated += selected_damage_gain_score
        combined_gain_accumulated += selected_combined_gain
        timing_record['selected_general_gain'] = float(selected_general_gain)
        timing_record['selected_damage_gain'] = float(selected_damage_gain)
        timing_record['selected_damage_gain_score'] = float(selected_damage_gain_score)
        timing_record['selected_combined_gain'] = float(selected_combined_gain)
        if damage_enabled:
            selected_damage_coverage = (
                best_beam['damage_coverage_history'][0]
                if len(best_beam['damage_coverage_history']) > 0
                else current_damage_coverage
            )
            damage_coverage_curve.append(float(selected_damage_coverage))
            timing_record['damage_coverage'] = float(selected_damage_coverage)
            damage_metric_rows.append({
                'step': int(pose_i),
                'general_coverage': float(coverage_evolution[-1]) if len(coverage_evolution) > 0 else 0.0,
                'damage_coverage': float(selected_damage_coverage),
                'general_gain': float(selected_general_gain),
                'damage_gain': float(selected_damage_gain),
                'damage_gain_score': float(selected_damage_gain_score),
                'combined_gain': float(selected_combined_gain),
            })

        stage_start = _profile_now(device)
        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            if transmission_camera is not None:
                transmission_camera.update_camera(next_idx, interpolation_step=interpolation_step)
                capture_transmitted_highres_frame(mesh, camera, transmission_camera)
            else:
                camera.capture_image(mesh)
            interpolation_step += 1
        timing_record['action_execution_time_s'] = _profile_now(device) - stage_start
        timing_record['total_iteration_time_s'] = _profile_now(device) - iteration_start
        timing_records.append(timing_record)

        pose_i += 1

    damage_metrics = None
    if damage_enabled:
        final_damage_coverage = damage_coverage_curve[-1] if len(damage_coverage_curve) > 0 else 0.0
        final_general_coverage = coverage_evolution[-1] if len(coverage_evolution) > 0 else 0.0
        damage_metrics = {
            'final_damage_coverage': float(final_damage_coverage),
            'damage_coverage_auc': compute_damage_coverage_auc(damage_coverage_curve),
            'final_general_coverage': float(final_general_coverage),
            'damage_gain_accumulated': float(damage_gain_accumulated),
            'damage_gain_score_accumulated': float(damage_gain_score_accumulated),
            'combined_gain_accumulated': float(combined_gain_accumulated),
            'general_gain_accumulated': float(selected_general_gain_accumulated),
            'final_damage_multiview_coverage': float(final_damage_coverage),
            'damage_coverage_curve': damage_coverage_curve,
            'damage_metrics_rows': damage_metric_rows,
        }
        if damage_config['save_damage_metrics']:
            write_damage_metrics_csv(
                os.path.join(damage_debug_dir, 'damage_metrics.csv'),
                damage_metric_rows,
            )
        print(
            "Final damage-aware metrics: "
            f"final_damage_coverage={final_damage_coverage:.4f}, "
            f"final_general_coverage={final_general_coverage:.4f}, "
            f"damage_gain_accumulated={damage_gain_accumulated:.2f}, "
            f"damage_gain_score_accumulated={damage_gain_score_accumulated:.2f}, "
            f"combined_gain_accumulated={combined_gain_accumulated:.2f}"
        )

    print("Coverage Evolution:", coverage_evolution)
    
    return (
        coverage_evolution,
        camera.X_cam_history,
        camera.V_cam_history,
        full_pc,
        full_pc_colors,
        full_pc_idx,
        timing_records,
        communication_records,
        damage_regions,
        damage_config if damage_enabled else None,
        damage_metrics,
        latest_damage_weight_stats,
    )
        
def run_magician_test(params_name,
             model_name,
             results_json_name,
             numGPU,
             test_scenes,
             test_resolution=0.05,
             use_perfect_depth_map=False,
             compute_collision=False,
             load_json=False,
             dataset_path=None,
             test_params=None):

    params_path = os.path.join(configs_dir, params_name)
    weights_path = os.path.join(weights_dir, model_name)
    results_json_path = os.path.join(results_dir, results_json_name)

    params = load_params(params_path)
    params.test_scenes = test_scenes
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
    params.anomaly_detection = False
    params.memory_dir_name = "test_memory_" + str(numGPU)
    if test_params is not None and hasattr(test_params, 'memory_dir_name'):
        params.memory_dir_name = test_params.memory_dir_name

    params.jz = False
    params.numGPU = numGPU
    params.WORLD_SIZE = 1
    params.batch_size = 1
    params.total_batch_size = 1

    if dataset_path is None:
        params.data_path = data_path
    else:
        params.data_path = dataset_path

    run_random_seed = int(getattr(test_params, 'random_seed', 0)) if test_params is not None else 0
    run_torch_seed = int(getattr(test_params, 'torch_seed', run_random_seed)) if test_params is not None else run_random_seed
    np.random.seed(run_random_seed)
    torch.manual_seed(run_torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_torch_seed)

    # Setup device
    device = setup_device(params, None)

    # Setup model and dataloader
    dataloader, macarons, memory = setup_test(params, weights_path, device)

    params.beam_width = test_params.beam_width
    params.beam_steps = test_params.beam_steps
    if hasattr(test_params, 'n_interpolation_steps'):
        params.n_interpolation_steps = test_params.n_interpolation_steps
    params.planning_image_height = getattr(test_params, 'planning_image_height', params.image_height)
    params.planning_image_width = getattr(test_params, 'planning_image_width', params.image_width)
    params.reconstruction_image_height = getattr(
        test_params, 'reconstruction_image_height', params.planning_image_height
    )
    params.reconstruction_image_width = getattr(
        test_params, 'reconstruction_image_width', params.planning_image_width
    )
    params.transmit_high_resolution = getattr(test_params, 'transmit_high_resolution', False)
    params.r_aux = float(getattr(test_params, 'r_aux', 1.0))
    params.degrade_mode = getattr(test_params, 'degrade_mode', 'block')
    params.degrade_seed = int(getattr(test_params, 'degrade_seed', 0))
    params.degrade_block_size = int(getattr(test_params, 'degrade_block_size', 16))
    params.image_height = params.planning_image_height
    params.image_width = params.planning_image_width
    params.export_hda_roi_dataset = bool(getattr(test_params, 'export_hda_roi_dataset', False))
    params.hda_roi_export_root = getattr(test_params, 'hda_roi_export_root', None)
    params.hda_export_dataset_id = getattr(
        test_params,
        'hda_export_dataset_id',
        getattr(test_params, 'hda_roi_damage_config_id', 'observations'),
    )
    params.hda_roi_damage_config_id = getattr(test_params, 'hda_roi_damage_config_id', params.hda_export_dataset_id)
    params.hda_roi_save_depth = bool(getattr(test_params, 'hda_roi_save_depth', True))
    params.hda_roi_save_valid_mask = bool(getattr(test_params, 'hda_roi_save_valid_mask', True))
    params.hda_roi_skip_existing_trajectories = bool(
        getattr(test_params, 'hda_roi_skip_existing_trajectories', False)
    )
    params.max_trajectories_per_scene = getattr(test_params, 'max_trajectories_per_scene', None)
    damage_defaults = _get_damage_config(test_params)
    for key, value in damage_defaults.items():
        setattr(params, key, value)

    lmdb_dir = os.path.join(results_dir, test_params.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"\nLMDB database directory: {lmdb_dir}")
    print(f"Results JSON path: {results_json_path}")

    results_records = []

    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict['occupied_pose']]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            # segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
           
            mesh_for_check = trimesh.load(mesh_path)

            if isinstance(mesh_for_check, trimesh.Scene):
                mesh_for_check = mesh_for_check.dump(concatenate=True)
            mesh_for_check.vertices *= params.scene_scale_factor

            intersector = mesh_for_check.ray

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)

            torch.cuda.empty_cache()

            max_trajectories_per_scene = getattr(params, 'max_trajectories_per_scene', None)
            scene_start_positions = _build_scene_start_positions(
                settings=settings,
                occupied_pose_data=occupied_pose_data,
                requested_count=max_trajectories_per_scene,
                device=device,
                scene_name=scene_name,
            )
            n_start_positions = len(scene_start_positions)

            scene_damage_regions_cache = None
            if (
                params.damage_aware_planning
                and params.export_hda_roi_dataset
                and getattr(params, 'damage_seed_scope', 'scene') == 'scene'
                and params.hda_roi_export_root is not None
            ):
                scene_damage_regions_cache = _load_hda_roi_scene_damage_regions(
                    params.hda_roi_export_root,
                    scene_name,
                    params.hda_roi_damage_config_id,
                )
                if scene_damage_regions_cache is not None:
                    print(
                        "Loaded existing scene-level HDA ROI damage regions: "
                        f"{scene_name}/{params.hda_roi_damage_config_id}"
                    )
            for start_cam_idx_i in range(n_start_positions):
                start_cam_idx = scene_start_positions[start_cam_idx_i]
                print("\n" + "="*60)
                print(f"Start cam index {start_cam_idx_i} for {scene_name}: {start_cam_idx}")
                print("="*60)

                # Each start_cam_idx_i gets its own trajectory number
                trajectory_nb = start_cam_idx_i
                training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
                _ensure_trajectory_memory_dirs(training_frames_path)
                print(f"Using trajectory folder: {training_frames_path}")
                trajectory_export_dir = None
                if params.export_hda_roi_dataset:
                    if params.hda_roi_export_root is None:
                        raise ValueError(
                            "export_hda_roi_dataset=True requires hda_roi_export_root in the test config."
                        )
                    trajectory_export_dir = os.path.join(
                        params.hda_roi_export_root,
                        'scenes',
                        scene_name,
                        params.hda_export_dataset_id,
                        f"traj{start_cam_idx_i}",
                    )
                    if (
                        params.hda_roi_skip_existing_trajectories
                        and _hda_roi_trajectory_export_complete(
                            trajectory_export_dir,
                            save_depth=params.hda_roi_save_depth,
                            save_valid_mask=params.hda_roi_save_valid_mask,
                        )
                    ):
                        print(f"Skipping existing HDA ROI trajectory export: {trajectory_export_dir}")
                        continue

                # Setup the Scene and Camera objects
                gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
                gc.collect()
                torch.cuda.empty_cache()
                gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                       mesh,
                                                                                       settings,
                                                                                       mirrored_scene,
                                                                                       device,
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                transmission_camera = None
                # clear_folder(training_frames_path)
                camera = setup_test_camera(params, mesh, intersector, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis,
                                           capture_initial=not params.transmit_high_resolution)
                highres_frames_path = None
                if params.transmit_high_resolution:
                    trajectory_dir = os.path.dirname(training_frames_path)
                    highres_frames_path = os.path.join(trajectory_dir, 'frames_highres')
                    os.makedirs(highres_frames_path, exist_ok=True)
                    transmission_camera = setup_test_camera(
                        params, mesh, intersector, start_cam_idx, settings, occupied_pose_data,
                        device, highres_frames_path,
                        mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis,
                        image_height=params.reconstruction_image_height,
                        image_width=params.reconstruction_image_width,
                        capture_initial=False,
                    )
                    if params.export_hda_roi_dataset:
                        transmission_camera.hda_roi_export_context = {
                            'trajectory_dir': trajectory_export_dir,
                            'scene_id': scene_name,
                            'damage_enabled': bool(params.damage_aware_planning),
                            'damage_config_id': params.hda_roi_damage_config_id,
                            'trajectory_id': f"traj{start_cam_idx_i}",
                            'save_depth': params.hda_roi_save_depth,
                            'save_valid_mask': params.hda_roi_save_valid_mask,
                        }
                    capture_transmitted_highres_frame(mesh, camera, transmission_camera)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                (
                    coverage_evolution,
                    X_cam_history,
                    V_cam_history,
                    full_pc,
                    full_pc_colors,
                    full_pc_idx,
                    timing_records,
                    communication_records,
                    damage_regions,
                    damage_config,
                    damage_metrics,
                    damage_weight_stats_record,
                ) = compute_magician_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      intersector,
                                                                                      device,
                                                                                      settings,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision,
                                                                                      transmission_camera=transmission_camera,
                                                                                      scene_name=scene_name,
                                                                                      trajectory_id=start_cam_idx_i,
                                                                                      preset_damage_regions=scene_damage_regions_cache)
                if (
                    params.damage_aware_planning
                    and damage_regions is not None
                    and getattr(params, 'damage_seed_scope', 'scene') == 'scene'
                    and scene_damage_regions_cache is None
                ):
                    scene_damage_regions_cache = damage_regions
                

                # Open LMDB, save data, then close
                print(f"\n=== Saving trajectory data to LMDB ===")
                lmdb_env = lmdb.open(lmdb_dir, map_size=30 * 1024 * 1024 * 1024)

                # Save trajectory data to LMDB
                lmdb_key = f"{scene_name}/{start_cam_idx_i}"
                trajectory_data = {
                    'coverage': coverage_evolution,
                    'X_cam_history': X_cam_history.cpu().numpy(),
                    'V_cam_history': V_cam_history.cpu().numpy(),
                    'points': full_pc.cpu().numpy(),
                    'points_color': full_pc_colors.cpu().numpy(),
                    'timing': timing_records,
                    'communication': communication_records,
                    'n_interpolation_steps': params.n_interpolation_steps,
                    'planning_resolution': (params.planning_image_height, params.planning_image_width),
                    'reconstruction_resolution': (
                        params.reconstruction_image_height,
                        params.reconstruction_image_width,
                    ),
                    'transmit_high_resolution': params.transmit_high_resolution,
                    'planning_frames_path': training_frames_path,
                    'highres_frames_path': highres_frames_path,
                }
                if params.damage_aware_planning:
                    trajectory_data.update({
                        'damage_regions': damage_regions,
                        'damage_config': damage_config,
                        'damage_metrics': damage_metrics,
                        'damage_weight_stats': damage_weight_stats_record,
                    })
                save_to_lmdb(lmdb_env, lmdb_key, trajectory_data)

                # Close LMDB
                lmdb_env.close()
                print(f"Closed LMDB database for {scene_name}/{start_cam_idx_i}\n")

                # Cleanup: Keep only imgs folder, delete frames/depths/occupancy folders
                # cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs'])
                # print(f"Finished processing trajectory {start_cam_idx_i}\n")

                result_record = {
                    'scene': scene_name,
                    'trajectory_id': int(start_cam_idx_i),
                    'start_cam_idx': _to_jsonable(start_cam_idx),
                    'coverage': coverage_evolution,
                    'final_general_coverage': coverage_evolution[-1] if len(coverage_evolution) > 0 else 0.0,
                    'lmdb_key': lmdb_key,
                    'timing': timing_records,
                    'communication': communication_records,
                }
                if params.damage_aware_planning:
                    result_record.update({
                        'damage_regions': damage_regions,
                        'damage_config': damage_config,
                        'damage_metrics': damage_metrics,
                        'damage_weight_stats': damage_weight_stats_record,
                    })
                results_records.append(result_record)
                os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
                with open(results_json_path, 'w') as outfile:
                    json.dump(_to_jsonable({'trajectories': results_records}), outfile, indent=2)
                print(f"Saved trajectory summary JSON to {results_json_path}")

    print("All trajectories computed.")
