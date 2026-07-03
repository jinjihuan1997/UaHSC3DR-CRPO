"""Build a sample-level offline lookup table for DRL simulation.

Each CSV row is one real sample under one (SNR, k_d) condition. It contains
RGB JSCC quality and the corresponding digital depth delivery metadata.
"""
import argparse
import csv
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import HDAROIDataset
from src.models.hda_semcom import HDAROISystem
from src.channel.channels import digital_link_budget_from_snr, analog_noise_std
from src.utils.metrics import region_psnr


def normalize_traj_id(value):
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("traj"):
        return text
    return f"traj{int(float(text))}"


def parse_traj_ids(value):
    if value is None:
        return None
    return {normalize_traj_id(x) for x in value.split(",") if x.strip()}


def build_dataset(cfg, split, scene_id_filter=None, manifest_rel=None):
    d = cfg["data"]
    manifest = manifest_rel or {"train": d["train_manifest"], "val": d["val_manifest"], "test": d["test_manifest"]}[split]
    return HDAROIDataset(
        root=d["root"],
        manifest_rel=manifest,
        roi_source=d["roi_source"],
        image_size=d["image_size"],
        soft_mask_blur=d.get("soft_mask_blur", 9),
        mode="train" if split == "train" else "val",
        use_original_resolution=d.get("use_original_resolution", False),
        aux_payload_source=d.get("aux_payload_source", "zlib_crop"),
        aux_payload_size=d.get("aux_payload_size"),
        min_rgb_std=d.get("min_rgb_std", 1e-4),
        scene_id_filter=(scene_id_filter if scene_id_filter is not None
                         else d.get("train_scene_id") if split == "train" else None),
    )


def tensor_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().view(-1)[0].item())
    return float(value)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/offline_lookup_table.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default=None,
                    help="Optional manifest path relative to data.root, overriding --split manifest.")
    ap.add_argument("--scene-id", default=None,
                    help="Only use samples from this scene_id. Omit to use all scenes in the split.")
    ap.add_argument("--trajectory-ids", default=None,
                    help="Comma-separated trajectories to include, e.g. 0,1,2 or traj0,traj1.")
    ap.add_argument("--snrs", type=float, nargs="+", default=None)
    ap.add_argument("--digital-subcarriers", type=int, nargs="+", default=None)
    ap.add_argument("--sample-indices", type=int, nargs="+", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    device = cfg["train"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ds = build_dataset(cfg, args.split, scene_id_filter=args.scene_id, manifest_rel=args.manifest)
    traj_ids = parse_traj_ids(args.trajectory_ids)
    if args.sample_indices is not None:
        indices = [i for i in args.sample_indices if 0 <= i < len(ds)]
    else:
        indices = list(range(len(ds)))
    if traj_ids is not None:
        indices = [
            i for i in indices
            if normalize_traj_id(ds.samples[i].get("trajectory_id", "")) in traj_ids
        ]
    if args.max_samples is not None:
        indices = indices[:args.max_samples]

    # The offline RL table should match the semantic-communication training
    # resource grid by default. CLI flags can still override this for sweeps.
    snrs = (args.snrs if args.snrs is not None
            else cfg["channel"].get("train_snr_points",
                                    cfg.get("eval", {}).get("snr_points", [10.0])))
    k_grid = args.digital_subcarriers or cfg["channel"].get("digital_subcarrier_grid", [12])
    total_data_subcarriers = int(cfg["channel"].get("data_subcarriers", 24))

    model = HDAROISystem(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "sample_idx", "scene_id", "trajectory_id", "frame_id", "clean_image",
        "snr_db", "k_d", "k_rgb", "rgb_symbols", "digital_re", "mcs",
        "depth_payload_bits", "depth_bit_budget", "depth_success", "depth_success_rate",
        "rgb_psnr", "valid_psnr", "invalid_psnr", "noise_std",
        "load_latency_ms", "forward_latency_ms",
    ]

    total_rows = len(indices) * len(snrs) * len(k_grid)
    written = 0
    start = time.perf_counter()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_pos, sample_idx in enumerate(indices, start=1):
            load_t0 = time.perf_counter()
            sample = ds[sample_idx]
            load_latency_ms = (time.perf_counter() - load_t0) * 1000.0
            rec = ds.samples[sample_idx]

            rgb = sample["rgb"].unsqueeze(0).to(device)
            depth = sample["depth"].unsqueeze(0).to(device)
            valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
            depth_bits = sample["depth_payload_bits"].view(1).to(device)

            for k_d in k_grid:
                cfg["channel"]["resource_allocation"] = "fixed_subcarriers"
                cfg["channel"]["digital_resource_ratio"] = int(k_d) / float(total_data_subcarriers)
                model.cfg = cfg

                for snr in snrs:
                    snr = float(snr)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = model(rgb, depth, valid_mask, snr, training=False, aux_payload_bits=depth_bits)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    forward_latency_ms = (time.perf_counter() - t0) * 1000.0

                    full, valid, invalid = region_psnr(rgb, out["rgb_hat"], valid_mask)
                    budget = digital_link_budget_from_snr(snr, cfg, device=device)
                    payload = tensor_scalar(depth_bits)
                    depth_budget = tensor_scalar(out["aux_info"]["aux_budget_bits"])
                    depth_success_rate = 1.0 if payload <= 0 else min(1.0, depth_budget / payload)
                    depth_success = float(depth_success_rate >= 1.0)
                    mcs = int(tensor_scalar(budget["mcs_index"]))

                    writer.writerow({
                        "sample_idx": sample_idx,
                        "scene_id": rec.get("scene_id", ""),
                        "trajectory_id": rec.get("trajectory_id", ""),
                        "frame_id": rec.get("frame_id", rec.get("frame", "")),
                        "clean_image": rec.get("clean_image", ""),
                        "snr_db": snr,
                        "k_d": int(k_d),
                        "k_rgb": int(budget["analog_data_subcarriers"]),
                        "rgb_symbols": int(tensor_scalar(budget["analog_symbols"])),
                        "digital_re": int(tensor_scalar(budget["digital_resource_elements"])),
                        "mcs": "OUT" if mcs < 0 else mcs,
                        "depth_payload_bits": payload,
                        "depth_bit_budget": depth_budget,
                        "depth_success": depth_success,
                        "depth_success_rate": depth_success_rate,
                        "rgb_psnr": full,
                        "valid_psnr": valid,
                        "invalid_psnr": invalid,
                        "noise_std": float(analog_noise_std(snr).item()),
                        "load_latency_ms": load_latency_ms,
                        "forward_latency_ms": forward_latency_ms,
                    })
                    written += 1

            if sample_pos == 1 or sample_pos % 10 == 0 or sample_pos == len(indices):
                elapsed = time.perf_counter() - start
                print(
                    f"[{sample_pos}/{len(indices)} samples] wrote {written}/{total_rows} rows "
                    f"elapsed={elapsed:.1f}s out={args.out}",
                    flush=True,
                )

    print(f"Wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
