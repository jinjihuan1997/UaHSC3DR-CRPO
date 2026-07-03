"""Evaluate CRPO-inspired PPO and allocation baselines."""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel.channels import default_mcs_table
from src.rl.cmdp_resource_env import CMDPResourceAllocationEnv
from src.rl.crpo_ppo import load_policy
from src.rl.mapping_quality import gsfusion_reconstruction_surrogate
from src.rl.resource_model import (
    action_to_resource,
    constraint_costs,
    depth_success_rate,
    split_snr_from_link_gain,
)
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


def normalize_traj_id(value):
    text = str(value).strip()
    if text.startswith("traj"):
        return text
    return f"traj{int(float(text))}"


def write_csv(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def closest_index(values, target):
    return int(np.argmin([abs(float(v) - float(target)) for v in values]))


def make_oracle_action(lambda_r, lambda_d):
    """Per-slot enumeration policy over the full discrete action space.

    At each slot it evaluates every (k_D, beta_D) pair with the same
    transmission model used by the environment and picks the action that
    maximizes q_3d - lambda_r * c_R - lambda_d * c_D. With
    lambda_r = lambda_d = 0 this is an unconstrained per-slot greedy
    oracle; with positive multipliers it is a per-slot Lagrangian policy.
    It uses privileged information (the instantaneous link gain and the
    per-action lookup outcome before committing to an action).
    """
    def choose(e, _obs):
        best_action = (0, 0)
        best_score = -float("inf")
        for ik in range(len(e.k_d_choices)):
            for ib in range(len(e.beta_d_choices)):
                resource = action_to_resource(
                    (ik, ib), e.k_d_choices, e.beta_d_choices, e.k_total)
                snr_info = split_snr_from_link_gain(
                    e.current["link_gain"],
                    resource["k_d"],
                    resource["beta_d"],
                    e.k_total,
                    p_total_watt=e.p_total_watt,
                    bandwidth_hz=e.bandwidth_hz,
                    n0_watt_per_hz=e.n0_watt_per_hz,
                    noise_figure_linear=e.noise_figure_linear,
                )
                q_rgb, _, _ = e._lookup_rgb_quality(
                    e.current["sample_idx"], snr_info["snr_rgb_db"],
                    resource["k_d"], resource["k_rgb"])
                depth_info = depth_success_rate(
                    snr_info["snr_depth_db"],
                    resource["k_d"],
                    e.current["depth_payload_bits"],
                    n_depth_blocks=e.n_depth_blocks,
                    slot_duration_s=e.slot_duration_s,
                    ofdm_symbol_duration_us=e.ofdm_symbol_duration_us,
                    communication_resource_fraction=e.communication_resource_fraction,
                    mac_efficiency=e.mac_efficiency,
                    mcs_table=e.mcs_table,
                )
                quality_info = gsfusion_reconstruction_surrogate(
                    q_rgb=q_rgb,
                    r_depth=depth_info["r_depth"],
                    view_importance=1.0,
                    map_progress=e.map_progress,
                    cfg=e.mapping_quality_cfg,
                )
                c_r, c_d = constraint_costs(
                    q_rgb, depth_info["r_depth"], e.q_req, e.depth_req)
                score = (float(quality_info["q_3d"])
                         - lambda_r * float(c_r) - lambda_d * float(c_d))
                if score > best_score:
                    best_score = score
                    best_action = (ik, ib)
        return np.array(best_action)
    return choose


def run_policy(env, choose_action, episodes):
    episode_returns = []
    episode_cost_rgb = []
    episode_cost_depth = []
    final_q_3d = []
    q_3d = []
    q_rgb = []
    r_depth = []
    reconstruction_gain = []
    cost_rgb = []
    cost_depth = []
    sat_rgb = []
    sat_depth = []
    snr_rgb = []
    snr_depth = []
    k_d = []
    k_rgb = []
    beta_d = []
    payload = []
    budget = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        ep_cost_rgb = []
        ep_cost_depth = []
        while not done:
            action = choose_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            q_3d.append(float(info["q_3d"]))
            q_rgb.append(float(info["q_rgb"]))
            r_depth.append(float(info["r_depth"]))
            reconstruction_gain.append(float(info["reconstruction_gain"]))
            cost_rgb.append(float(info["cost_rgb"]))
            cost_depth.append(float(info["cost_depth"]))
            ep_cost_rgb.append(float(info["cost_rgb"]))
            ep_cost_depth.append(float(info["cost_depth"]))
            sat_rgb.append(float(info["cost_rgb"] <= 1e-8))
            sat_depth.append(float(info["cost_depth"] <= 1e-8))
            snr_rgb.append(float(info["snr_rgb_db"]))
            snr_depth.append(float(info["snr_depth_db"]))
            k_d.append(float(info["k_d"]))
            k_rgb.append(float(info["k_rgb"]))
            beta_d.append(float(info["beta_d"]))
            payload.append(float(info["depth_payload_bits"]))
            budget.append(float(info["depth_bit_budget"]))
        episode_returns.append(float(episode_return))
        episode_cost_rgb.append(float(np.mean(ep_cost_rgb)))
        episode_cost_depth.append(float(np.mean(ep_cost_depth)))
        final_q_3d.append(float(env.map_progress_final))
    return {
        "avg_map_progress": float(np.mean(final_q_3d)),   # episode-cumulative avg q_3d (≈ mean q_3d over episode)
        "avg_reconstruction_surrogate_reward": float(np.mean(q_3d)),
        "avg_q3d": float(np.mean(q_3d)),
        "avg_q_3d": float(np.mean(q_3d)),
        "avg_map_reward": float(np.mean(q_3d)),
        "avg_q_rgb": float(np.mean(q_rgb)),
        "avg_r_depth": float(np.mean(r_depth)),
        "avg_reconstruction_gain": float(np.mean(reconstruction_gain)),
        "J_C_R": float(np.mean(episode_cost_rgb)),
        "J_C_D": float(np.mean(episode_cost_depth)),
        "avg_cost_rgb": float(np.mean(cost_rgb)),
        "avg_cost_depth": float(np.mean(cost_depth)),
        "rgb_violation_rate": float(np.mean(np.asarray(cost_rgb) > 1e-8)),
        "depth_violation_rate": float(np.mean(np.asarray(cost_depth) > 1e-8)),
        "constraint_satisfaction_rgb": float(np.mean(sat_rgb)),
        "constraint_satisfaction_depth": float(np.mean(sat_depth)),
        "avg_snr_rgb_db": float(np.mean(snr_rgb)),
        "avg_snr_depth_db": float(np.mean(snr_depth)),
        "avg_kd": float(np.mean(k_d)),
        "avg_k_d": float(np.mean(k_d)),
        "avg_k_rgb": float(np.mean(k_rgb)),
        "avg_beta_d": float(np.mean(beta_d)),
        "avg_depth_payload_bits": float(np.mean(payload)),
        "avg_depth_bit_budget": float(np.mean(budget)),
        "episode_return": float(np.mean(episode_returns)),
    }


def aggregate_rows(rows):
    numeric_keys = [
        "avg_reconstruction_surrogate_reward", "avg_q3d", "avg_q_rgb", "avg_r_depth",
        "J_C_R", "J_C_D", "rgb_violation_rate",
        "depth_violation_rate", "avg_kd", "avg_beta_d", "episode_return",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, group in sorted(grouped.items()):
        agg = {
            "method": method,
            "trajectory_id": "aggregate",
            "shadow_seed": "aggregate",
            "action_seed": "aggregate",
        }
        for key in numeric_keys:
            agg[key] = float(np.mean([float(row[key]) for row in group]))
        out.append(agg)
    return out


def make_env(args, cfg, mapping_cfg, ckpt, k_choices, beta_choices, trajectory_ids, seed):
    ch = cfg.get("channel", {})
    channel_kwargs = {
        "fc_hz": args.fc_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "ptx_dbm": args.ptx_dbm,
        "noise_figure_db": args.noise_figure_db,
        "shadow_sigma_db": args.shadow_sigma_db,
    }
    return CMDPResourceAllocationEnv(
        args.table,
        episode_len=args.episode_len,
        seed=seed,
        trajectory_csv=args.trajectory_csv,
        channel_kwargs=channel_kwargs,
        trajectory_reset=args.trajectory_reset,
        trajectory_ids=trajectory_ids,
        k_total=int(args.k_total or ckpt.get("k_total", 24)),
        k_d_choices=k_choices,
        beta_d_choices=beta_choices,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--model", "--checkpoint", dest="model", required=True, help="CRPO-PPO .pt checkpoint.")
    ap.add_argument("--ppo-penalty-model", default=None, help="Optional PPO-penalty .pt checkpoint.")
    ap.add_argument("--ppo-penalty-label", default="PPO-penalty", help="Label for the PPO-penalty method in output.")
    ap.add_argument("--lagrangian-model", default=None, help="Optional Lagrangian-PPO .pt checkpoint.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--episode-len", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--trajectory-csv", default=None)
    ap.add_argument("--trajectory-reset", default="trajectory", choices=["random", "zero", "trajectory"])
    ap.add_argument("--multi-trajectory", action="store_true",
                    help="Evaluate each selected trajectory separately.")
    ap.add_argument("--eval-split", default="test", choices=["train", "val", "test"],
                    help="Trajectory split to evaluate when --test-trajs/--eval-trajs is omitted.")
    ap.add_argument("--test-trajs", "--eval-trajs", dest="test_trajs", default=None,
                    help="Comma-separated trajectories to evaluate, e.g. 7,8 or 9,10.")
    ap.add_argument("--shadow-seeds", default=None,
                    help="Comma-separated shadowing seeds. Defaults to --seed when omitted.")
    ap.add_argument("--random-action-seeds", default=None,
                    help="Comma-separated action seeds for the random allocation baseline.")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated methods to evaluate. Valid names: random,fixed-balanced,RGB-priority,depth-priority,CRPO-PPO,PPO-penalty,Lagrangian-PPO.")
    ap.add_argument("--per-traj-csv", default=None)
    ap.add_argument("--aggregate-csv", default=None)
    ap.add_argument("--fc-hz", type=float, default=900e6)
    ap.add_argument("--bandwidth-hz", type=float, default=1e6)
    ap.add_argument("--ptx-dbm", type=float, default=20.0)
    ap.add_argument("--noise-figure-db", type=float, default=7.0)
    ap.add_argument("--shadow-sigma-db", type=float, default=3.0)
    ap.add_argument("--k-total", type=int, default=24)
    ap.add_argument("--k-d-choices", default=None)
    ap.add_argument("--beta-d-choices", default=None)
    ap.add_argument("--q-req", type=float, default=0.6)
    ap.add_argument("--depth-req", type=float, default=0.8)
    ap.add_argument("--n-depth-blocks", type=int, default=None)
    ap.add_argument("--mapping-quality-mode", default=None, choices=["gsfusion_surrogate"])
    ap.add_argument("--per-trajectory-surrogate-yaml", default=None,
                    help="YAML with per-trajectory surrogate weights.")
    ap.add_argument("--psnr-ref-min", type=float, default=None,
                    help="Reference PSNR min for normalization (p5 of full table).")
    ap.add_argument("--psnr-ref-max", type=float, default=None,
                    help="Reference PSNR max for normalization (p95 of full table).")
    ap.add_argument("--payload-ref-min", type=float, default=None,
                    help="Reference depth payload min (p5 of full table).")
    ap.add_argument("--payload-ref-max", type=float, default=None,
                    help="Reference depth payload max (p95 of full table).")
    ap.add_argument("--oracle-lambda-r", type=float, default=0.0,
                    help="RGB-cost multiplier for the per-slot-lagrangian method.")
    ap.add_argument("--oracle-lambda-d", type=float, default=0.0,
                    help="Depth-cost multiplier for the per-slot-lagrangian method.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    ch = cfg.get("channel", {})
    mapping_cfg = dict(cfg.get("mapping_quality", {}))
    if args.mapping_quality_mode:
        mapping_cfg["mapping_quality_mode"] = args.mapping_quality_mode
    policy, ckpt = load_policy(args.model, device=args.device)
    penalty_policy = None
    penalty_ckpt = None
    if args.ppo_penalty_model:
        penalty_policy, penalty_ckpt = load_policy(args.ppo_penalty_model, device=args.device)
    lagrangian_policy = None
    lagrangian_ckpt = None
    if args.lagrangian_model:
        lagrangian_policy, lagrangian_ckpt = load_policy(args.lagrangian_model, device=args.device)
    k_choices = parse_csv_numbers(args.k_d_choices, int) or ckpt.get("k_d_choices")
    beta_choices = parse_csv_numbers(args.beta_d_choices, float) or ckpt.get("beta_d_choices")
    if len(k_choices) != int(ckpt.get("n_kd", len(k_choices))):
        raise ValueError("--k-d-choices length must match checkpoint action head")
    if len(beta_choices) != int(ckpt.get("n_beta", len(beta_choices))):
        raise ValueError("--beta-d-choices length must match checkpoint action head")
    if penalty_ckpt is not None:
        if len(k_choices) != int(penalty_ckpt.get("n_kd", len(k_choices))):
            raise ValueError("--k-d-choices length must match PPO-penalty checkpoint action head")
        if len(beta_choices) != int(penalty_ckpt.get("n_beta", len(beta_choices))):
            raise ValueError("--beta-d-choices length must match PPO-penalty checkpoint action head")
    if lagrangian_ckpt is not None:
        if len(k_choices) != int(lagrangian_ckpt.get("n_kd", len(k_choices))):
            raise ValueError("--k-d-choices length must match Lagrangian-PPO checkpoint action head")
        if len(beta_choices) != int(lagrangian_ckpt.get("n_beta", len(beta_choices))):
            raise ValueError("--beta-d-choices length must match Lagrangian-PPO checkpoint action head")

    def crpo_action(_env, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            action, _, _, _ = policy.act(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    def penalty_action(_env, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            action, _, _, _ = penalty_policy.act(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    def lagrangian_action(_env, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            action, _, _, _ = lagrangian_policy.act(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    selected_methods = None
    if args.methods:
        selected_methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    valid_methods = {
        "random", "fixed-balanced", "RGB-priority", "depth-priority",
        "CRPO-PPO", "PPO-penalty", "Lagrangian-PPO",
        "greedy-oracle", "per-slot-lagrangian",
    }
    if selected_methods:
        unknown = selected_methods - valid_methods
        if unknown:
            raise ValueError(f"unknown method(s): {sorted(unknown)}")

    def include_method(method):
        return selected_methods is None or method in selected_methods

    def make_random_action(seed):
        rng = random.Random(int(seed))
        def choose(e, _obs):
            return np.array([
                rng.randrange(len(e.k_d_choices)),
                rng.randrange(len(e.beta_d_choices)),
            ])
        return choose

    def method_actions(env, random_action_seed=None):
        fixed_k_idx = closest_index(env.k_d_choices, env.k_total / 2.0)
        fixed_beta_idx = closest_index(env.beta_d_choices, 0.5)
        rgb_k_idx = closest_index(env.k_d_choices, min(env.k_d_choices))
        rgb_beta_idx = closest_index(env.beta_d_choices, min(env.beta_d_choices))
        depth_k_idx = closest_index(env.k_d_choices, max(env.k_d_choices))
        depth_beta_idx = closest_index(env.beta_d_choices, max(env.beta_d_choices))
        methods = {}
        if include_method("random"):
            methods["random"] = make_random_action(args.seed if random_action_seed is None else random_action_seed)
        if include_method("fixed-balanced"):
            methods["fixed-balanced"] = lambda _e, _obs: np.array([fixed_k_idx, fixed_beta_idx])
        if include_method("RGB-priority"):
            methods["RGB-priority"] = lambda _e, _obs: np.array([rgb_k_idx, rgb_beta_idx])
        if include_method("depth-priority"):
            methods["depth-priority"] = lambda _e, _obs: np.array([depth_k_idx, depth_beta_idx])
        if include_method("CRPO-PPO"):
            methods["CRPO-PPO"] = crpo_action
        if include_method("PPO-penalty") and penalty_policy is not None:
            methods[args.ppo_penalty_label] = penalty_action
        if include_method("Lagrangian-PPO") and lagrangian_policy is not None:
            methods["Lagrangian-PPO"] = lagrangian_action
        # Oracle methods run only when explicitly selected via --methods.
        if selected_methods and "greedy-oracle" in selected_methods:
            methods["greedy-oracle"] = make_oracle_action(0.0, 0.0)
        if selected_methods and "per-slot-lagrangian" in selected_methods:
            methods["per-slot-lagrangian"] = make_oracle_action(
                args.oracle_lambda_r, args.oracle_lambda_d)
        return methods

    eval_jobs = [(None, args.seed)]
    if args.multi_trajectory:
        seeds = parse_csv_numbers(args.shadow_seeds, int) or [args.seed]
        eval_trajs = args.test_trajs or ",".join(str(x) for x in default_split_trajs(args.eval_split))
        eval_jobs = [
            (normalize_traj_id(traj), int(seed))
            for traj in parse_csv_numbers(eval_trajs, str)
            for seed in seeds
        ]

    random_action_seeds = parse_csv_numbers(args.random_action_seeds, int) or [args.seed]

    rows = []
    for traj_id, seed in eval_jobs:
        prototype_env = make_env(
            args, cfg, mapping_cfg, ckpt, k_choices, beta_choices,
            [traj_id] if traj_id is not None else None,
            seed,
        )
        method_seed_pairs = []
        if include_method("random"):
            for action_seed in random_action_seeds:
                method_seed_pairs.append(("random", action_seed))
        for method in ["fixed-balanced", "RGB-priority", "depth-priority", "CRPO-PPO", "PPO-penalty", "Lagrangian-PPO"]:
            if include_method(method):
                method_seed_pairs.append((method, None))
        for method in ["greedy-oracle", "per-slot-lagrangian"]:
            if selected_methods and method in selected_methods:
                method_seed_pairs.append((method, None))
        for method_key, action_seed in method_seed_pairs:
            actions = method_actions(prototype_env, action_seed)
            if method_key == "PPO-penalty":
                output_method = args.ppo_penalty_label
            else:
                output_method = method_key
            if output_method not in actions:
                continue
            choose_action = actions[output_method]
            env = make_env(
                args, cfg, mapping_cfg, ckpt, k_choices, beta_choices,
                [traj_id] if traj_id is not None else None,
                seed,
            )
            metrics = run_policy(env, choose_action, args.episodes)
            rows.append({
                "method": output_method,
                "trajectory_id": traj_id or "all",
                "shadow_seed": seed,
                "action_seed": action_seed if action_seed is not None else "",
                "avg_reconstruction_surrogate_reward": metrics["avg_reconstruction_surrogate_reward"],
                "avg_q3d": metrics["avg_q3d"],
                "avg_q_rgb": metrics["avg_q_rgb"],
                "avg_r_depth": metrics["avg_r_depth"],
                "J_C_R": metrics["J_C_R"],
                "J_C_D": metrics["J_C_D"],
                "rgb_violation_rate": metrics["rgb_violation_rate"],
                "depth_violation_rate": metrics["depth_violation_rate"],
                "avg_kd": metrics["avg_kd"],
                "avg_beta_d": metrics["avg_beta_d"],
                "episode_return": metrics["episode_return"],
            })

    aggregate = aggregate_rows(rows)
    write_csv(args.per_traj_csv, rows)
    write_csv(args.aggregate_csv, aggregate)

    print(f"table={args.table}")
    print(f"episodes={args.episodes} episode_len={args.episode_len} q_req={args.q_req} depth_req={args.depth_req}")
    header = (
        f"{'method':<18} {'traj':>8} {'seed':>6} {'aseed':>6} {'avg3d':>8} {'Qrgb':>7} {'Rdep':>7} "
        f"{'J_C_R':>7} {'J_C_D':>7} {'violR':>7} {'violD':>7} {'kd':>6} {'beta':>6} {'return':>8}"
    )
    print(header)
    for row in rows + aggregate:
        print(
            f"{row['method']:<18} {row['trajectory_id']:>8} {str(row['shadow_seed']):>6} "
            f"{str(row.get('action_seed', '')):>6} "
            f"{row['avg_q3d']:>8.3f} {row['avg_q_rgb']:>7.3f} {row['avg_r_depth']:>7.3f} "
            f"{row['J_C_R']:>7.3f} {row['J_C_D']:>7.3f} "
            f"{row['rgb_violation_rate']:>7.3f} {row['depth_violation_rate']:>7.3f} "
            f"{row['avg_kd']:>6.2f} {row['avg_beta_d']:>6.3f} {row['episode_return']:>8.3f}"
        )


if __name__ == "__main__":
    main()
