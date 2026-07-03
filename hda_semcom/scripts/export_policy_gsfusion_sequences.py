"""Export policy-generated RGB-D test sequences for method-level GSFusion evaluation."""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_gsfusion_degraded_sequences import (  # noqa: E402
    _build_dataset,
    _c2w_from_pose,
    _dataset_index_by_frame_key,
    _degrade_depth_blocks,
    _intrinsics_from_pose,
    _load_depth_u16,
    _normalize_traj_id,
    _pose_json,
    _resolve,
    _stable_depth_seed,
    _write_gsfusion_config,
)
from scripts.eval_crpo_ppo import closest_index, parse_csv_numbers  # noqa: E402
from src.channel.channels import default_mcs_table  # noqa: E402
from src.models.hda_semcom import HDAROISystem  # noqa: E402
from src.rl.cmdp_resource_env import CMDPResourceAllocationEnv  # noqa: E402
from src.rl.crpo_ppo import load_policy  # noqa: E402
from src.utils.config import load_config  # noqa: E402


METHOD_DISPLAY = {
    "fixed-balanced": "Fixed-balanced allocation",
    "RGB-priority": "RGB-priority allocation",
    "depth-priority": "Depth-priority allocation",
    "random": "Random allocation",
    "PPO-penalty": "PPO-penalty",
    "Lagrangian-PPO": "Lagrangian-PPO",
    "CRPO-PPO": "CRPO-guided PPO",
}


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _training_seed_from_tag(tag):
    match = re.search(r"trainseed(\d+)", str(tag or ""))
    return int(match.group(1)) if match else None


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_env(args, cfg, mapping_cfg, ckpt, k_choices, beta_choices, traj_id, seed):
    ch = cfg.get("channel", {})
    return CMDPResourceAllocationEnv(
        args.table,
        episode_len=args.episode_len,
        seed=seed,
        trajectory_csv=args.trajectory_csv,
        channel_kwargs={
            "fc_hz": args.fc_hz,
            "bandwidth_hz": args.bandwidth_hz,
            "ptx_dbm": args.ptx_dbm,
            "noise_figure_db": args.noise_figure_db,
            "shadow_sigma_db": args.shadow_sigma_db,
        },
        trajectory_reset="trajectory",
        trajectory_ids=[traj_id],
        k_total=int(args.k_total or ckpt.get("k_total", 24)),
        k_d_choices=k_choices,
        beta_d_choices=beta_choices,
        q_req=args.q_req,
        depth_req=args.depth_req,
        slot_duration_s=float(ch.get("slot_duration_s", 0.5)),
        ofdm_symbol_duration_us=float(ch.get("ofdm_symbol_duration_us", 40.0)),
        communication_resource_fraction=float(ch.get("communication_resource_fraction", 0.8)),
        mac_efficiency=float(ch.get("mac_efficiency", 0.8)),
        n_depth_blocks=args.n_depth_blocks,
        mcs_table=ch.get("mcs_table", default_mcs_table()),
        bandwidth_hz=args.bandwidth_hz,
        ptx_dbm=args.ptx_dbm,
        noise_figure_db=args.noise_figure_db,
        mapping_quality_mode=mapping_cfg.get("mapping_quality_mode", "gsfusion_surrogate"),
        mapping_quality_cfg=mapping_cfg,
        per_trajectory_surrogate_yaml=args.per_trajectory_surrogate_yaml,
        psnr_ref_min=args.psnr_ref_min,
        psnr_ref_max=args.psnr_ref_max,
        payload_ref_min=args.payload_ref_min,
        payload_ref_max=args.payload_ref_max,
    )


def _policy_action(policy, device):
    def choose(_env, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = policy.act(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy()
    return choose


def _method_actions(args, env, crpo_policy, penalty_policy, lagrangian_policy):
    fixed_k_idx = closest_index(env.k_d_choices, env.k_total / 2.0)
    fixed_beta_idx = closest_index(env.beta_d_choices, 0.5)
    rgb_k_idx = closest_index(env.k_d_choices, min(env.k_d_choices))
    rgb_beta_idx = closest_index(env.beta_d_choices, min(env.beta_d_choices))
    depth_k_idx = closest_index(env.k_d_choices, max(env.k_d_choices))
    depth_beta_idx = closest_index(env.beta_d_choices, max(env.beta_d_choices))

    methods = {
        "fixed-balanced": lambda _e, _obs: np.array([fixed_k_idx, fixed_beta_idx]),
        "RGB-priority": lambda _e, _obs: np.array([rgb_k_idx, rgb_beta_idx]),
        "depth-priority": lambda _e, _obs: np.array([depth_k_idx, depth_beta_idx]),
        "random": None,
    }
    if crpo_policy is not None:
        methods["CRPO-PPO"] = _policy_action(crpo_policy, args.device)
    if penalty_policy is not None:
        methods["PPO-penalty"] = _policy_action(penalty_policy, args.device)
    if lagrangian_policy is not None:
        methods["Lagrangian-PPO"] = _policy_action(lagrangian_policy, args.device)
    return methods


def _collect_episode(env, choose_action, method_key, seed, random_action_seed=None):
    obs, _ = env.reset(seed=seed)
    done = False
    rows = []
    rng_seed = seed + (zlib_crc(method_key) % 1000003)
    if method_key == "random" and random_action_seed is not None:
        rng_seed = int(random_action_seed)
    rng = np.random.default_rng(rng_seed)
    while not done:
        if choose_action is None:
            action = np.array([
                int(rng.integers(0, len(env.k_d_choices))),
                int(rng.integers(0, len(env.beta_d_choices))),
            ])
        else:
            action = choose_action(env, obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(dict(info))
    return rows


def zlib_crc(text):
    import zlib
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _dataset_sample(ds, frame_key_to_idx, scene_id, traj_id, frame_id):
    key = (scene_id, _normalize_traj_id(traj_id), int(frame_id))
    if key not in frame_key_to_idx:
        raise KeyError(f"frame not found in dataset manifest: {key}")
    sample_idx = frame_key_to_idx[key]
    return sample_idx, ds[sample_idx], ds.samples[sample_idx]


@torch.no_grad()
def _export_condition(args, cfg, model, ds, frame_key_to_idx, method_key, traj_id, seed, policy_rows, random_action_seed=None):
    method_display = METHOD_DISPLAY.get(method_key, method_key)
    tag = f"{_slug(args.condition_tag)}_" if getattr(args, "condition_tag", "") else ""
    cond_name = f"{tag}{_slug(method_display)}_{traj_id}_chseed{seed}"
    if method_key == "random" and random_action_seed is not None:
        cond_name += f"_actionseed{int(random_action_seed)}"
    cond_dir = Path(args.out) / cond_name
    seq_dir = cond_dir / "sequence"
    results_dir = seq_dir / "results"
    output_dir = cond_dir / "gsfusion_output"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    gsfusion_root = Path(args.gsfusion_root).resolve()
    optim_params = gsfusion_root / "parameter/optimization_params_degraded.json"
    c2w_list = []
    frame_meta = []
    first_pose = None
    width = height = None
    depth_scale = 1.0 / 6553.5

    for local_idx, info in enumerate(policy_rows[:args.episode_len]):
        sample_idx, sample, rec = _dataset_sample(
            ds,
            frame_key_to_idx,
            args.scene_id,
            info["trajectory_id"],
            info["frame_id"],
        )
        pose = _pose_json(cfg["data"]["root"], rec)
        if first_pose is None:
            first_pose = pose
        depth_scale = float(pose.get("depth_encoding", {}).get("scale", depth_scale))

        rgb = sample["rgb"].unsqueeze(0).to(args.device)
        depth = sample["depth"].unsqueeze(0).to(args.device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(args.device)
        depth_bits = sample["depth_payload_bits"].view(1).to(args.device)

        cfg["channel"]["resource_allocation"] = "fixed_subcarriers"
        cfg["channel"]["digital_resource_ratio"] = int(info["k_d"]) / float(args.k_total)
        model.cfg = cfg
        out = model(
            rgb,
            depth,
            valid_mask,
            float(info["snr_rgb_db"]),
            training=False,
            aux_payload_bits=depth_bits,
        )
        rgb_hat = out["rgb_hat"].detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb_u8 = (rgb_hat * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(rgb_u8).save(results_dir / f"frame{local_idx:06d}.jpg", quality=95)

        depth_path = Path(ds._resolve(rec, "depth"))
        depth_u16 = _load_depth_u16(depth_path)
        if rgb_u8.shape[:2] != depth_u16.shape[:2]:
            depth_u16 = np.asarray(
                Image.fromarray(depth_u16).resize((rgb_u8.shape[1], rgb_u8.shape[0]), Image.NEAREST)
            )
        degraded_depth = _degrade_depth_blocks(
            depth_u16,
            info["r_depth"],
            args.n_depth_blocks,
            seed=_stable_depth_seed(cond_name, local_idx),
        )
        Image.fromarray(degraded_depth).save(results_dir / f"depth{local_idx:06d}.png")

        c2w = _c2w_from_pose(pose)
        c2w_list.append(c2w)
        height, width = rgb_u8.shape[:2]
        frame_meta.append({
            "local_frame": local_idx,
            "sample_idx": sample_idx,
            "scene_id": rec.get("scene_id", args.scene_id),
            "trajectory_id": rec.get("trajectory_id", traj_id),
            "frame_id": rec.get("frame_id", info["frame_id"]),
            "clean_image": str(_resolve(cfg["data"]["root"], rec, "clean_image")),
            "depth_image": str(depth_path),
            "method": method_display,
            "q_rgb": float(info["q_rgb"]),
            "r_depth": float(info["r_depth"]),
            "k_d": int(info["k_d"]),
            "k_rgb": int(info["k_rgb"]),
            "beta_d": float(info["beta_d"]),
            "total_snr_db": float(info["total_snr_db"]),
            "snr_rgb_db": float(info["snr_rgb_db"]),
            "snr_depth_db": float(info["snr_depth_db"]),
            "depth_payload_bits": float(info["depth_payload_bits"]),
            "depth_bit_budget": float(info["depth_bit_budget"]),
        })

    if args.no_normalize_poses or not c2w_list:
        local_c2w_list = c2w_list
    else:
        base_inv = np.linalg.inv(c2w_list[0])
        local_c2w_list = [base_inv @ c2w for c2w in c2w_list]
    (seq_dir / "traj.txt").write_text(
        "\n".join(" ".join(f"{v:.9g}" for v in c2w.reshape(-1)) for c2w in local_c2w_list) + "\n"
    )
    intrinsics = _intrinsics_from_pose(first_pose or {}, width, height)
    config_path = cond_dir / "gsfusion_config.yaml"
    _write_gsfusion_config(
        config_path,
        seq_dir.resolve(),
        output_dir.resolve(),
        optim_params.resolve(),
        width,
        height,
        intrinsics,
        depth_scale,
        len(frame_meta),
    )
    meta = {
        "condition": cond_name,
        "method": method_display,
        "method_key": method_key,
        "trajectory_id": traj_id,
        "training_seed": _training_seed_from_tag(getattr(args, "condition_tag", "")),
        "shadow_seed": seed,
        "action_seed": int(random_action_seed) if random_action_seed is not None else None,
        "q_rgb": float(np.mean([m["q_rgb"] for m in frame_meta])),
        "r_depth": float(np.mean([m["r_depth"] for m in frame_meta])),
        "avg_k_d": float(np.mean([m["k_d"] for m in frame_meta])),
        "avg_beta_d": float(np.mean([m["beta_d"] for m in frame_meta])),
        "config_path": str(config_path.resolve()),
        "sequence_path": str(seq_dir.resolve()),
        "output_path": str(output_dir.resolve()),
        "frames": frame_meta,
    }
    (cond_dir / "condition_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="outputs/eiffel15_surrogate_test/lookups/test_eiffel.csv")
    ap.add_argument("--trajectory-csv", default="outputs/eiffel15_surrogate_test/trajectory_eiffel_test.csv")
    ap.add_argument("--test-trajs", default="traj10,traj11,traj12,traj13,traj14")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--semcom-ckpt", default="checkpoints/stage3_final.pt")
    ap.add_argument("--model", default="checkpoints/crpo_eiffel_composite_app60202_q070_d035.pt")
    ap.add_argument("--ppo-penalty-model", default="checkpoints/penalty_eiffel_composite_app60202_q070_d035.pt")
    ap.add_argument("--lagrangian-model", default="checkpoints/lagrangian_eiffel_composite_app60202_q070_d035.pt")
    ap.add_argument("--out", default="outputs/eiffel15_mapping_quality/conditions")
    ap.add_argument("--gsfusion-root", default="/home/king/Downloads/Projects/TCOM/GSFusion")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default="manifests/eiffel15_surrogate_test/test_eiffel.jsonl")
    ap.add_argument("--scene-id", default="eiffel")
    ap.add_argument("--episode-len", type=int, default=102)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--random-action-seed", type=int, default=None,
                    help="Independent action-sampling seed for the random allocation baseline.")
    ap.add_argument("--condition-tag", default="",
                    help="Optional prefix tag inserted into exported condition names, e.g. trainseed0.")
    ap.add_argument("--append-index", action="store_true",
                    help="Append to an existing conditions_index.json and per_slot_policy.csv instead of replacing them.")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--methods", default="fixed-balanced,RGB-priority,depth-priority,random,PPO-penalty,Lagrangian-PPO,CRPO-PPO")
    ap.add_argument("--fc-hz", type=float, default=900e6)
    ap.add_argument("--bandwidth-hz", type=float, default=1e6)
    ap.add_argument("--ptx-dbm", type=float, default=20.0)
    ap.add_argument("--noise-figure-db", type=float, default=7.0)
    ap.add_argument("--shadow-sigma-db", type=float, default=4.0)
    ap.add_argument("--k-total", type=int, default=24)
    ap.add_argument("--k-d-choices", default="10,12,14,16,18,20,22")
    ap.add_argument("--beta-d-choices", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--q-req", type=float, default=0.70)
    ap.add_argument("--depth-req", type=float, default=0.35)
    ap.add_argument("--n-depth-blocks", type=int, default=16)
    ap.add_argument("--per-trajectory-surrogate-yaml", default="outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.yaml")
    ap.add_argument("--psnr-ref-min", type=float, default=8.9171)
    ap.add_argument("--psnr-ref-max", type=float, default=22.5791)
    ap.add_argument("--payload-ref-min", type=float, default=78344.0)
    ap.add_argument("--payload-ref-max", type=float, default=644408.0)
    ap.add_argument("--no-normalize-poses", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    ds = _build_dataset(cfg, args.split, manifest_rel=args.manifest, scene_id_filter=args.scene_id)
    frame_key_to_idx = _dataset_index_by_frame_key(ds)

    model = HDAROISystem(cfg).to(args.device)
    model.load_state_dict(torch.load(args.semcom_ckpt, map_location=args.device))
    model.eval()

    selected_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    known_methods = set(METHOD_DISPLAY)
    unknown_methods = selected_methods - known_methods
    if unknown_methods:
        raise SystemExit(f"unknown method(s) in --methods: {sorted(unknown_methods)}")

    ckpt = {}
    crpo_policy = None
    if "CRPO-PPO" in selected_methods:
        crpo_policy, ckpt = load_policy(args.model, device=args.device)
    penalty_policy = (
        load_policy(args.ppo_penalty_model, device=args.device)[0]
        if "PPO-penalty" in selected_methods and args.ppo_penalty_model
        else None
    )
    lagrangian_policy = (
        load_policy(args.lagrangian_model, device=args.device)[0]
        if "Lagrangian-PPO" in selected_methods and args.lagrangian_model
        else None
    )
    k_choices = parse_csv_numbers(args.k_d_choices, int) or ckpt.get("k_d_choices")
    beta_choices = parse_csv_numbers(args.beta_d_choices, float) or ckpt.get("beta_d_choices")
    if not k_choices or not beta_choices:
        raise SystemExit("missing action choices; provide --k-d-choices and --beta-d-choices")
    mapping_cfg = dict(cfg.get("mapping_quality", {}))
    mapping_cfg["mapping_quality_mode"] = "gsfusion_surrogate"

    traj_ids = [_normalize_traj_id(x) for x in args.test_trajs.split(",") if x.strip()]
    all_conditions = []
    per_slot_rows = []

    for traj_id in traj_ids:
        prototype_env = _make_env(args, cfg, mapping_cfg, ckpt, k_choices, beta_choices, traj_id, args.seed)
        methods = _method_actions(args, prototype_env, crpo_policy, penalty_policy, lagrangian_policy)
        for method_key, choose_action in methods.items():
            if method_key not in selected_methods:
                continue
            env = _make_env(args, cfg, mapping_cfg, ckpt, k_choices, beta_choices, traj_id, args.seed)
            action_seed = args.random_action_seed if method_key == "random" else None
            policy_rows = _collect_episode(env, choose_action, method_key, args.seed, action_seed)
            if len(policy_rows) != args.episode_len:
                raise RuntimeError(
                    f"{method_key} on {traj_id} produced {len(policy_rows)} frames; "
                    f"expected --episode-len={args.episode_len}. Check trajectory completeness."
                )
            meta = _export_condition(args, cfg, model, ds, frame_key_to_idx, method_key, traj_id, args.seed, policy_rows, action_seed)
            all_conditions.append(meta)
            for t, row in enumerate(policy_rows):
                out = {
                    "condition": meta["condition"],
                    "method": meta["method"],
                    "trajectory_id": traj_id,
                    "training_seed": meta["training_seed"] if meta["training_seed"] is not None else "",
                    "shadow_seed": args.seed,
                    "action_seed": action_seed if action_seed is not None else "",
                    "slot": t,
                }
                out.update({k: row[k] for k in [
                    "frame_id", "k_d", "k_rgb", "beta_d", "total_snr_db", "snr_rgb_db",
                    "snr_depth_db", "q_rgb", "r_depth", "cost_rgb", "cost_depth",
                    "depth_payload_bits", "depth_bit_budget", "mcs",
                ] if k in row})
                per_slot_rows.append(out)
            print(f"[exported] {meta['condition']} frames={len(meta['frames'])}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.append_index:
        index_path = out_root / "conditions_index.json"
        if index_path.exists():
            all_conditions = json.loads(index_path.read_text()) + all_conditions
        per_slot_path = out_root / "per_slot_policy.csv"
        if per_slot_path.exists():
            with per_slot_path.open(newline="") as f:
                old_rows = list(csv.DictReader(f))
            per_slot_rows = old_rows + per_slot_rows
    (out_root / "conditions_index.json").write_text(json.dumps(all_conditions, indent=2))
    _write_csv(out_root / "per_slot_policy.csv", per_slot_rows)
    print(f"wrote {out_root / 'conditions_index.json'} conditions={len(all_conditions)}")
    print(f"wrote {out_root / 'per_slot_policy.csv'} rows={len(per_slot_rows)}")


if __name__ == "__main__":
    main()
