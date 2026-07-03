"""
Sweep discrete digital subcarrier allocations and SNR points.

The table variable is the integer number of data subcarriers assigned to the
depth+valid_mask digital auxiliary link. Internally this is converted to the
exact realizable ratio k / data_subcarriers for the channel model.

Each CSV row is one (digital_subcarriers, snr_db) point. RGB semantic JSCC uses
the remaining data subcarriers.

Usage:
    python scripts/sweep_subcarrier_snr.py \
        --config configs/default.yaml \
        --ckpt checkpoints/stage3_final.pt \
        --digital-subcarriers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
        --out outputs/subcarrier_snr_table.csv
"""
import argparse
import copy
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_dataloader
from src.models.hda_roi import HDAROISystem
from src.channel.channels import analog_noise_std, digital_link_budget_from_snr
from scripts.evaluate import build_lpips_metric, evaluate


def _default_subcarriers(total_data_subcarriers):
    return list(range(1, total_data_subcarriers))


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().float().view(-1)[0].item())
    return float(value)


def _as_int(value):
    return int(round(_as_float(value)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/subcarrier_snr_table.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--scene-id", default=None,
                    help="Only use samples from this scene_id. Omit to use all scenes in the split.")
    ap.add_argument("--digital-subcarriers", type=int, nargs="+", default=None,
                    help="Integer data subcarrier counts assigned to the depth+valid_mask digital link.")
    ap.add_argument("--snrs", type=float, nargs="+", default=None,
                    help="SNR points in dB. Defaults to eval.snr_points from config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    total_data_subcarriers = int(cfg["channel"].get("data_subcarriers", 24))
    digital_subcarriers_grid = args.digital_subcarriers or _default_subcarriers(total_data_subcarriers)
    for k in digital_subcarriers_grid:
        if k < 0 or k > total_data_subcarriers:
            raise ValueError(f"digital subcarriers must be in [0, {total_data_subcarriers}], got {k}")

    snrs = args.snrs if args.snrs is not None else cfg["eval"]["snr_points"]
    loader = build_dataloader(cfg, args.split, scene_id_filter=args.scene_id)
    state = torch.load(args.ckpt, map_location=device)
    lpips_metric = build_lpips_metric(device)

    rows = []
    for digital_subcarriers in digital_subcarriers_grid:
        requested_ratio = digital_subcarriers / float(total_data_subcarriers)
        cfg_r = copy.deepcopy(cfg)
        cfg_r["channel"]["resource_allocation"] = "fixed_subcarriers"
        cfg_r["channel"]["digital_resource_ratio"] = float(requested_ratio)
        cfg_r["channel"].pop("digital_data_subcarriers", None)

        model = HDAROISystem(cfg_r).to(device)
        model.load_state_dict(state)

        for snr in snrs:
            budget = digital_link_budget_from_snr(float(snr), cfg_r, device=device)
            digital_subcarriers = int(budget["digital_data_subcarriers"])
            rgb_subcarriers = int(budget["analog_data_subcarriers"])
            effective_ratio = float(budget["effective_digital_resource_ratio"])
            full, valid, invalid, msssim, lpips_value, aux_ok, payload, aux_budget = evaluate(
                model, loader, float(snr), device, lpips_metric)
            mcs = _as_int(budget["mcs_index"])
            mcs_label = "OUT" if mcs < 0 else str(mcs)
            row = {
                "requested_digital_ratio": float(requested_ratio),
                "effective_digital_ratio": effective_ratio,
                "digital_subcarriers": digital_subcarriers,
                "rgb_subcarriers": rgb_subcarriers,
                "snr_db": float(snr),
                "mcs": mcs_label,
                "phy_rate_mbps": _as_float(budget["phy_rate_mbps"]),
                "digital_re": _as_int(budget["digital_resource_elements"]),
                "rgb_symbols": _as_int(budget["analog_symbols"]),
                "aux_budget_bits": float(aux_budget),
                "aux_payload_bits": float(payload),
                "aux_ok": float(aux_ok),
                "noise_std": float(analog_noise_std(float(snr)).item()),
                "rgb_psnr": float(full),
                "valid_psnr": float(valid),
                "invalid_psnr": float(invalid),
                "ms_ssim_db": float(msssim),
                "lpips": float(lpips_value),
            }
            rows.append(row)
            print(
                f"r_req={requested_ratio:.3f} r_eff={effective_ratio:.3f} "
                f"k_d={digital_subcarriers:2d} k_rgb={rgb_subcarriers:2d} "
                f"snr={float(snr):5.1f} mcs={mcs_label:>3} aux_ok={aux_ok:4.2f} "
                f"rgb_sym={row['rgb_symbols']:6d} psnr={full:5.2f} lpips={lpips_value:.4f}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
