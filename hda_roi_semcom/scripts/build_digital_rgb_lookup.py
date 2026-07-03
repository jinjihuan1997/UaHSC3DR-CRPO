"""Build an all-digital RGB lookup table for the architecture ablation (C2).

For each row of an existing JSCC lookup table, replace the analog-JSCC
``rgb_psnr`` with the PSNR obtained by conventional separate source-channel
coding: the RGB frame is JPEG-compressed at the highest quality whose payload
fits the MCS bit budget of the RGB stream in that slot. If even the lowest
JPEG quality does not fit, the frame is treated as lost (cliff effect) and the
PSNR is set to 0, which maps to Q_rgb = 0 after normalization.

The MCS selection and bit-budget computation mirror
``src.rl.resource_model.depth_success_rate`` so that the digital RGB stream
uses exactly the same PHY abstraction as the depth auxiliary stream.
"""
import argparse
import io
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.resource_model import highest_mcs_supported_by_snr
from src.utils.config import load_config

LADDER = [("JPEG", q) for q in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)] + \
         [("WEBP", q) for q in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90)]


def jpeg_ladder(image_path):
    """Return list of (bits, psnr, tag) over a JPEG+WebP quality ladder."""
    img = Image.open(image_path).convert("RGB")
    ref = np.asarray(img, dtype=np.float64)
    ladder = []
    for fmt, q in LADDER:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=q)
        bits = 8 * buf.getbuffer().nbytes
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
        mse = float(np.mean((ref - dec) ** 2))
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)
        ladder.append((float(bits), float(psnr), f"{fmt}{q}"))
    return image_path, ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="Existing JSCC lookup CSV.")
    ap.add_argument("--out", required=True, help="Output digital lookup CSV.")
    ap.add_argument("--data-root", default=None,
                    help="Dataset root for clean_image paths (default: config data.root).")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mac-efficiency", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    mcs_table = cfg.get("channel", {}).get("mcs_table")

    df = pd.read_csv(args.table)
    frames = sorted(df["clean_image"].unique())
    paths = [os.path.join(data_root, f) for f in frames]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames missing, e.g. {missing[0]}")

    print(f"Encoding source-coding ladders for {len(paths)} frames "
          f"({len(LADDER)} rate points) with {args.workers} workers...")
    ladders = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (path, ladder) in enumerate(ex.map(jpeg_ladder, paths, chunksize=4)):
            ladders[os.path.relpath(path, data_root)] = ladder
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} frames done")

    out_psnr = np.zeros(len(df))
    out_bits = np.zeros(len(df))
    out_quality = np.full(len(df), "lost", dtype=object)
    out_budget = np.zeros(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        # Same PHY abstraction as depth_success_rate: budget = eta_MAC * N_RE * n_bpscs * r.
        mcs = highest_mcs_supported_by_snr(float(row.snr_db), mcs_table=mcs_table)
        if mcs is None:
            budget = 0.0
        else:
            budget = (float(row.rgb_symbols) * float(mcs["n_bpscs"])
                      * float(mcs["code_rate"]) * args.mac_efficiency)
        out_budget[i] = budget
        best = None
        for bits, psnr, q in ladders[row.clean_image]:
            if bits <= budget and (best is None or psnr > best[1]):
                best = (bits, psnr, q)
        if best is not None:
            out_bits[i], out_psnr[i], out_quality[i] = best
        # else: frame lost, rgb_psnr stays 0.0 (cliff)

    out = df.copy()
    out["rgb_psnr"] = out_psnr
    out["digital_rgb_bits"] = out_bits
    out["digital_rgb_jpeg_quality"] = out_quality
    out["digital_rgb_bit_budget"] = out_budget
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    lost = float(np.mean(out_psnr <= 0.0))
    print(f"Wrote {args.out}: {len(out)} rows, frame-loss ratio {lost:.3f}, "
          f"median delivered PSNR {np.median(out_psnr[out_psnr > 0]):.2f} dB")


if __name__ == "__main__":
    main()
