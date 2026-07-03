"""Train CRPO-inspired PPO for constrained resource allocation."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel.channels import default_mcs_table
from src.rl.cmdp_resource_env import CMDPResourceAllocationEnv
from src.rl.crpo_ppo import PPOConfig, train_crpo_ppo
from src.utils.config import load_config


def parse_csv_numbers(value, cast):
    if value is None:
        return None
    return [cast(x) for x in value.split(",") if x.strip()]


def default_split_trajs(split):
    return {
        "train": [0, 1, 2, 3, 4, 5, 6],
        "val": [7, 8],
        "test": [9, 10],
    }[split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="Offline lookup CSV for RGB quality and payload metadata.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="checkpoints/crpo_ppo_resource.pt")
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--timesteps", "--total-steps", dest="timesteps", type=int, default=100000)
    ap.add_argument("--run-name", default=None, help="Optional name used for default output/log paths.")
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--update-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--trajectory-csv", default=None)
    ap.add_argument("--trajectory-reset", default="trajectory", choices=["random", "zero", "trajectory"])
    ap.add_argument("--multi-trajectory", action="store_true",
                    help="Sample episodes from the train trajectory split traj0-traj6.")
    ap.add_argument("--trajectory-ids", default=None,
                    help="Comma-separated trajectories to use, e.g. 0,1,2 or traj0,traj1.")
    ap.add_argument("--fc-hz", type=float, default=900e6)
    ap.add_argument("--bandwidth-hz", type=float, default=1e6)
    ap.add_argument("--ptx-dbm", type=float, default=20.0)
    ap.add_argument("--noise-figure-db", type=float, default=7.0)
    ap.add_argument("--shadow-sigma-db", type=float, default=3.0)
    ap.add_argument("--k-total", type=int, default=24)
    ap.add_argument("--k-d-choices", default="10,12,14,16,18,20,22")
    ap.add_argument("--beta-d-choices", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--q-req", type=float, default=0.6)
    ap.add_argument("--depth-req", type=float, default=0.8)
    ap.add_argument("--epsilon-rgb", "--d-rgb", dest="epsilon_rgb", type=float, default=0.05)
    ap.add_argument("--epsilon-depth", "--d-depth", dest="epsilon_depth", type=float, default=0.05)
    ap.add_argument("--n-depth-blocks", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--mapping-quality-mode", default=None, choices=["gsfusion_surrogate"])
    ap.add_argument("--per-trajectory-surrogate-yaml", default=None,
                    help="YAML with per-trajectory surrogate weights. Overrides config gsfusion_weights per episode.")
    ap.add_argument("--objective", default="crpo", choices=["crpo", "penalty", "lagrangian"],
                    help="crpo: mode-switching; penalty: fixed-lambda PPO; lagrangian: adaptive-lambda PPO.")
    ap.add_argument("--penalty-rgb", type=float, default=1.0)
    ap.add_argument("--penalty-depth", type=float, default=1.0)
    ap.add_argument("--lr-dual", type=float, default=0.005,
                    help="Dual variable learning rate for Lagrangian objective.")
    ap.add_argument("--psnr-ref-min", type=float, default=None,
                    help="Reference PSNR min for normalization (p5 of full table). Ensures train/test consistency.")
    ap.add_argument("--psnr-ref-max", type=float, default=None,
                    help="Reference PSNR max for normalization (p95 of full table).")
    ap.add_argument("--payload-ref-min", type=float, default=None,
                    help="Reference depth payload min (p5 of full table).")
    ap.add_argument("--payload-ref-max", type=float, default=None,
                    help="Reference depth payload max (p95 of full table).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ch = cfg.get("channel", {})
    mapping_cfg = dict(cfg.get("mapping_quality", {}))
    if args.mapping_quality_mode:
        mapping_cfg["mapping_quality_mode"] = args.mapping_quality_mode
    if args.run_name and args.out == "checkpoints/crpo_ppo_resource.pt":
        args.out = os.path.join("checkpoints", f"{args.run_name}.pt")
    trajectory_ids = parse_csv_numbers(args.trajectory_ids, str)
    if args.multi_trajectory and trajectory_ids is None:
        trajectory_ids = default_split_trajs("train")
    channel_kwargs = {
        "fc_hz": args.fc_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "ptx_dbm": args.ptx_dbm,
        "noise_figure_db": args.noise_figure_db,
        "shadow_sigma_db": args.shadow_sigma_db,
    }
    log_csv = args.log_csv or os.path.join(
        "outputs", "ppo_logs", os.path.splitext(os.path.basename(args.out))[0], "crpo_ppo_log.csv")
    env = CMDPResourceAllocationEnv(
        args.table,
        episode_len=args.episode_len,
        seed=args.seed,
        trajectory_csv=args.trajectory_csv,
        channel_kwargs=channel_kwargs,
        trajectory_reset=args.trajectory_reset,
        trajectory_ids=trajectory_ids,
        k_total=args.k_total,
        k_d_choices=parse_csv_numbers(args.k_d_choices, int),
        beta_d_choices=parse_csv_numbers(args.beta_d_choices, float),
        q_req=args.q_req,
        depth_req=args.depth_req,
        slot_duration_s=float(ch.get("slot_duration_s", 0.5)),
        ofdm_symbol_duration_us=float(ch.get("ofdm_symbol_duration_us", 40.0)),
        communication_resource_fraction=float(ch.get("communication_resource_fraction", 0.8)),
        mac_efficiency=float(ch.get("mac_efficiency", 0.8)),
        n_depth_blocks=args.n_depth_blocks or int(mapping_cfg.get("n_depth_blocks", 16)),
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
    ppo_cfg = PPOConfig(
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epsilon_rgb=args.epsilon_rgb,
        epsilon_depth=args.epsilon_depth,
        hidden_dim=args.hidden_dim,
        objective=args.objective,
        penalty_rgb=args.penalty_rgb,
        penalty_depth=args.penalty_depth,
        lr_dual=args.lr_dual,
    )
    train_crpo_ppo(env, ppo_cfg, seed=args.seed, device=args.device, log_csv=log_csv, save_path=args.out)
    print(f"[saved] log: {log_csv}")


if __name__ == "__main__":
    main()
