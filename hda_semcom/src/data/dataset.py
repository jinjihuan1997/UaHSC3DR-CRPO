"""
读取 hda_semcom_dataset 的 jsonl 清单，配对 clean_image / roi_mask / roi_priority_map。

manifest 每行是一个 JSON 对象，含相对路径字段：
    clean_image, roi_mask, roi_priority_map, overlay, depth, camera_pose
本数据加载只用到前三个；路径相对于 data.root。
"""
import os
import io
import glob
import json
import random
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def _gaussian_blur_mask(mask_np: np.ndarray, ksize: int) -> np.ndarray:
    """对单通道 [0,1] 掩码做高斯模糊（不依赖 cv2，用可分离高斯核手写卷积）。"""
    if ksize <= 1:
        return mask_np
    if ksize % 2 == 0:
        ksize += 1
    sigma = ksize / 6.0
    ax = np.arange(ksize) - ksize // 2
    k1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    pad = ksize // 2
    m = np.pad(mask_np, pad, mode="edge")
    # 横向
    tmp = np.zeros_like(mask_np, dtype=np.float32)
    for i, w in enumerate(k1d):
        tmp += w * m[pad:pad + mask_np.shape[0], i:i + mask_np.shape[1]]
    m2 = np.pad(tmp, pad, mode="edge")
    out = np.zeros_like(mask_np, dtype=np.float32)
    for i, w in enumerate(k1d):
        out += w * m2[i:i + mask_np.shape[0], pad:pad + mask_np.shape[1]]
    return np.clip(out, 0.0, 1.0)


class HDAROIDataset(Dataset):
    """
    返回 (image, mask)：
        image: (3, H, W) float in [0,1]
        mask:  (1, H, W) float in [0,1]   ROI 区域≈1，背景≈0
    """

    def __init__(self, root, manifest_rel, roi_source="priority",
                 image_size=256, soft_mask_blur=9, mode="train",
                 input_modalities=None, use_original_resolution=False,
                 aux_payload_source="zlib_crop", aux_payload_size=None,
                 scene_id_filter=None, min_rgb_std=1e-4):
        self.root = root
        self.roi_source = roi_source
        self.image_size = image_size
        self.soft_mask_blur = soft_mask_blur
        self.mode = mode
        self.use_original_resolution = bool(use_original_resolution)
        self.aux_payload_source = aux_payload_source
        self.aux_payload_size = aux_payload_size
        self.scene_id_filter = scene_id_filter
        self.min_rgb_std = float(min_rgb_std or 0.0)

        manifest_path = os.path.join(root, manifest_rel)
        self.samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.scene_id_filter and rec.get("scene_id") != self.scene_id_filter:
                    continue
                self.samples.append(rec)
        if self.min_rgb_std > 0:
            before = len(self.samples)
            self.samples = [rec for rec in self.samples if self._rgb_std_ok(rec)]
            skipped = before - len(self.samples)
            if skipped:
                print(
                    f"[dataset] skipped {skipped} low-variance RGB samples "
                    f"(std < {self.min_rgb_std:g}) from {manifest_path}"
                )
        if len(self.samples) == 0:
            suffix = f" for scene_id={self.scene_id_filter}" if self.scene_id_filter else ""
            raise RuntimeError(f"Empty manifest: {manifest_path}{suffix}")

    def __len__(self):
        return len(self.samples)

    def _path_variants(self, rel):
        variants = [rel]
        if "damage_config" in rel:
            variants.append(rel.replace("damage_config_0001", "observations"))
            variants.append(rel.replace("damage_config", "observations"))
        if "observations" in rel:
            variants.append(rel.replace("observations", "damage_config_0001"))
        out = []
        for item in variants:
            if item and item not in out:
                out.append(item)
        return out

    def _resolve(self, rec, *keys):
        """Resolve paths across legacy damage_config_* and new observations layouts."""
        for k in keys:
            rel = rec.get(k)
            if not rel:
                continue
            scene_id = rec.get("scene_id")
            trajectory_id = rec.get("trajectory_id")
            for cand in self._path_variants(rel):
                path = os.path.join(self.root, cand)
                if os.path.exists(path):
                    return path
                if scene_id and trajectory_id:
                    matches = glob.glob(os.path.join(
                        self.root, "scenes", scene_id, "*", trajectory_id, cand))
                    if matches:
                        return sorted(matches)[0]
            return os.path.join(self.root, rel)
        return None

    def _rgb_std_ok(self, rec):
        path = self._resolve(rec, "clean_image")
        if path is None or not os.path.exists(path):
            return True
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return float(arr.std()) >= self.min_rgb_std

    def _load_mask(self, rec, target_size):
        """
        加载 ROI 掩码。
        priority 模式优先用 roi_priority_map（软，连续 [0,1]）；
        mask 模式用 roi_mask（二值）。
        若所需文件缺失则回退到另一个；都缺失则全 1（退化为标准 HDA-DeepSC）。
        """
        if self.roi_source == "priority":
            path = self._resolve(rec, "roi_priority_map", "roi_mask")
        else:
            path = self._resolve(rec, "roi_mask", "roi_priority_map")

        if path is None or not os.path.exists(path):
            return Image.new("L", target_size, 255)
        return Image.open(path).convert("L")

    def _load_depth(self, rec, target_size):
        path = self._resolve(rec, "depth")
        if path is None or not os.path.exists(path):
            return Image.new("F", target_size, 0.0)
        return Image.open(path).convert("F")

    def _load_valid_mask(self, rec, target_size):
        path = self._resolve(rec, "valid_mask")
        if path is None or not os.path.exists(path):
            return Image.new("L", target_size, 255)
        return Image.open(path).convert("L")

    def _crop_size(self):
        if isinstance(self.image_size, (list, tuple)):
            return int(self.image_size[0]), int(self.image_size[1])
        size = int(self.image_size)
        return size, size

    def _aux_size(self):
        if self.aux_payload_size is None:
            return self._crop_size()
        return int(self.aux_payload_size[0]), int(self.aux_payload_size[1])

    @staticmethod
    def _png_bits(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return len(buf.getvalue()) * 8

    def _resized_png_payload_bits(self, depth_path, valid_mask_path):
        aux_h, aux_w = self._aux_size()
        target_size = (aux_w, aux_h)
        if depth_path and os.path.exists(depth_path):
            depth_img = Image.open(depth_path)
            depth_bits = self._png_bits(depth_img.resize(target_size, Image.NEAREST))
        else:
            depth_bits = 0
        if valid_mask_path and os.path.exists(valid_mask_path):
            mask_img = Image.open(valid_mask_path).convert("L")
            valid_mask_bits = self._png_bits(mask_img.resize(target_size, Image.NEAREST))
        else:
            valid_mask_bits = 0
        return depth_bits, valid_mask_bits

    @staticmethod
    def _depth_to_tensor(depth):
        arr = np.asarray(depth, dtype=np.float32)
        max_value = float(arr.max()) if arr.size else 0.0
        if max_value > 1.0:
            arr = arr / 65535.0
        return torch.from_numpy(np.clip(arr, 0.0, 1.0)).unsqueeze(0)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        img_path = self._resolve(rec, "clean_image")
        depth_path = self._resolve(rec, "depth")
        valid_mask_path = self._resolve(rec, "valid_mask")
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(
                f"clean_image missing for sample {idx}: {rec.get('clean_image')}")
        img = Image.open(img_path).convert("RGB")
        mask = self._load_mask(rec, img.size)
        depth = self._load_depth(rec, img.size)
        valid_mask = self._load_valid_mask(rec, img.size)

        # ---- 同步几何变换（图像、深度与掩码必须完全一致）----
        if not self.use_original_resolution:
            crop_h, crop_w = self._crop_size()
            if self.mode == "train":
                # 若原图小于裁剪尺寸，先等比放大
                if img.size[0] < crop_w or img.size[1] < crop_h:
                    scale = max(crop_w / img.size[0], crop_h / img.size[1])
                    new_size = (int(round(img.size[0] * scale)) + 1,
                                int(round(img.size[1] * scale)) + 1)
                    img = img.resize(new_size, Image.BICUBIC)
                    depth = depth.resize(new_size, Image.BILINEAR)
                    mask = mask.resize(new_size, Image.NEAREST)
                    valid_mask = valid_mask.resize(new_size, Image.NEAREST)
                i = random.randint(0, img.size[1] - crop_h)
                j = random.randint(0, img.size[0] - crop_w)
                box = (j, i, j + crop_w, i + crop_h)
                img = img.crop(box)
                depth = depth.crop(box)
                mask = mask.crop(box)
                valid_mask = valid_mask.crop(box)
                if random.random() > 0.5:
                    img = TF.hflip(img)
                    depth = TF.hflip(depth)
                    mask = TF.hflip(mask)
                    valid_mask = TF.hflip(valid_mask)
            else:
                img = TF.center_crop(img, [crop_h, crop_w])
                depth = TF.center_crop(depth, [crop_h, crop_w])
                mask = TF.center_crop(mask, [crop_h, crop_w])
                valid_mask = TF.center_crop(valid_mask, [crop_h, crop_w])
        elif self.mode == "train" and random.random() > 0.5:
            img = TF.hflip(img)
            depth = TF.hflip(depth)
            mask = TF.hflip(mask)
            valid_mask = TF.hflip(valid_mask)

        img_t = TF.to_tensor(img)  # (3,H,W) in [0,1]
        depth_t = self._depth_to_tensor(depth)

        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        if self.roi_source == "mask":
            mask_np = (mask_np > 0.5).astype(np.float32)  # 硬二值
        else:
            # priority 软掩码：保留连续值，可选高斯模糊缓解边界伪影
            mask_np = _gaussian_blur_mask(mask_np, self.soft_mask_blur)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)
        valid_mask_np = (np.asarray(valid_mask, dtype=np.float32) > 127.5).astype(np.float32)
        valid_mask_t = torch.from_numpy(valid_mask_np).unsqueeze(0)

        if self.aux_payload_source == "png_file_size":
            depth_bits = os.path.getsize(depth_path) * 8 if depth_path and os.path.exists(depth_path) else 0
            valid_mask_bits = (os.path.getsize(valid_mask_path) * 8
                               if valid_mask_path and os.path.exists(valid_mask_path) else 0)
        elif self.aux_payload_source == "resized_png_bytes":
            depth_bits, valid_mask_bits = self._resized_png_payload_bits(depth_path, valid_mask_path)
        else:
            depth_u16 = np.clip(depth_t.squeeze(0).numpy() * 65535.0, 0, 65535).astype(np.uint16)
            valid_u8 = valid_mask_t.squeeze(0).numpy().astype(np.uint8)
            depth_bits = len(zlib.compress(depth_u16.tobytes())) * 8
            valid_mask_bits = len(zlib.compress(np.packbits(valid_u8).tobytes())) * 8
        rgb_bits = os.path.getsize(img_path) * 8

        return {
            "rgb": img_t,
            "depth": depth_t,
            "valid_mask": valid_mask_t,
            "roi_mask": mask_t,
            "aux_payload_bits": torch.tensor(float(depth_bits + valid_mask_bits)),
            "depth_payload_bits": torch.tensor(float(depth_bits)),
            "depth_bits": torch.tensor(float(depth_bits)),
            "valid_mask_bits": torch.tensor(float(valid_mask_bits)),
            "rgb_png_bits": torch.tensor(float(rgb_bits)),
        }


def build_dataloader(cfg, split, scene_id_filter=None):
    from torch.utils.data import DataLoader
    d = cfg["data"]
    manifest = {"train": d["train_manifest"],
                "val": d["val_manifest"],
                "test": d["test_manifest"]}[split]
    ds = HDAROIDataset(
        root=d["root"], manifest_rel=manifest,
        roi_source=d["roi_source"], image_size=d["image_size"],
        soft_mask_blur=d.get("soft_mask_blur", 9),
        mode="train" if split == "train" else "val",
        use_original_resolution=d.get("use_original_resolution", False),
        aux_payload_source=d.get("aux_payload_source", "zlib_crop"),
        aux_payload_size=d.get("aux_payload_size"),
        min_rgb_std=d.get("min_rgb_std", 1e-4),
        scene_id_filter=(scene_id_filter if scene_id_filter is not None
                         else d.get("train_scene_id") if split == "train" else None),
    )
    bs = cfg["train"]["batch_size"] if split == "train" else cfg["eval"]["batch_size"]
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"),
                      num_workers=d["num_workers"], pin_memory=d["pin_memory"],
                      drop_last=(split == "train"))
