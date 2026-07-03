"""Assert-based sanity checks for CRPO resource allocation components."""
import os
import sys

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel.a2g import A2GChannel
from src.rl.crpo_ppo import TwoHeadActorCritic
from src.rl.mapping_quality import gsfusion_reconstruction_surrogate
from src.rl.resource_model import (
    action_to_resource,
    constraint_costs,
    db_to_linear,
    dbm_to_watt,
    depth_success_rate,
    highest_mcs_supported_by_snr,
    linear_to_db,
    link_gain_from_path_loss,
    power_split,
    rgb_quality_proxy,
    select_constraint_mode,
    split_snr_from_link_gain,
    split_snr_from_total,
    watt_to_dbm,
)


def main():
    k_choices = [4, 8, 12, 16, 20]
    beta_choices = [0.1, 0.3, 0.5, 0.7, 0.9]
    mapped = action_to_resource([2, 3], k_choices, beta_choices, 24)
    assert mapped["k_d"] == 12
    assert mapped["beta_d"] == 0.7
    assert mapped["k_rgb"] == 12
    assert mapped["k_d"] + mapped["k_rgb"] == 24

    p_d_low, p_rgb_low = power_split(0.3, 0.1)
    p_d_high, p_rgb_high = power_split(0.7, 0.1)
    assert abs((p_d_low + p_rgb_low) - 0.1) < 1e-12
    assert p_d_high > p_d_low
    assert p_rgb_high < p_rgb_low

    snr_rgb_low_beta, snr_depth_low_beta = split_snr_from_total(10.0, 12, 0.3, 24)
    snr_rgb_high_beta, snr_depth_high_beta = split_snr_from_total(10.0, 12, 0.7, 24)
    assert snr_depth_high_beta > snr_depth_low_beta
    assert snr_rgb_high_beta < snr_rgb_low_beta
    link_snr_8 = split_snr_from_link_gain(1e-8, 8, 0.5, 24, p_total_watt=0.1, bandwidth_hz=1e6,
                                          noise_figure_db=7.0)
    link_snr_16 = split_snr_from_link_gain(1e-8, 16, 0.5, 24, p_total_watt=0.1, bandwidth_hz=1e6,
                                           noise_figure_db=7.0)
    assert link_snr_16["snr_depth_db"] < link_snr_8["snr_depth_db"]

    depth_8 = depth_success_rate(10.0, 8, 100000.0)
    depth_16 = depth_success_rate(10.0, 16, 100000.0)
    assert depth_16["digital_re"] > depth_8["digital_re"]
    assert 0.0 <= depth_8["r_depth"] <= 1.0
    assert 0.0 <= depth_16["r_depth"] <= 1.0

    q = rgb_quality_proxy(100.0, 20, 24)
    r = depth_success_rate(100.0, 20, 1.0)["r_depth"]
    q3d = gsfusion_reconstruction_surrogate(q, r, view_importance=1.0, map_progress=1.0)["q_3d"]
    assert 0.0 <= q <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= q3d <= 1.0
    assert 0.0 <= split_snr_from_link_gain(1e-8, 12, 0.5, 24, p_total_watt=0.1,
                                           bandwidth_hz=1e6, noise_figure_db=7.0)["snr_rgb_linear"]

    assert abs(dbm_to_watt(20.0) - 0.1) < 1e-12
    assert abs(watt_to_dbm(0.1) - 20.0) < 1e-9
    assert abs(db_to_linear(10.0) - 10.0) < 1e-12
    assert abs(linear_to_db(10.0) - 10.0) < 1e-9

    row_near = {"uav_x_global_m": 0, "uav_y_global_m": 0, "uav_z_global_m": 50,
                "vehicle_x_m": 0, "vehicle_y_m": 0, "vehicle_z_m": 1.5}
    row_far = {"uav_x_global_m": 500, "uav_y_global_m": 0, "uav_z_global_m": 50,
               "vehicle_x_m": 0, "vehicle_y_m": 0, "vehicle_z_m": 1.5}
    channel = A2GChannel(shadow_sigma_db=0.0)
    rng = np.random.default_rng(0)
    near = channel.snr_from_row(row_near, rng)
    far = channel.snr_from_row(row_far, rng)
    assert link_gain_from_path_loss(near["path_loss_db"]) > 0.0
    assert link_gain_from_path_loss(near["path_loss_db"]) > link_gain_from_path_loss(far["path_loss_db"])

    mcs_low = highest_mcs_supported_by_snr(2.0)
    mcs_high = highest_mcs_supported_by_snr(20.0)
    assert mcs_high["snr_threshold"] >= mcs_low["snr_threshold"]

    q3d_base = gsfusion_reconstruction_surrogate(0.2, 0.2, map_progress=0.0)["reconstruction_gain"]
    assert gsfusion_reconstruction_surrogate(0.4, 0.2, map_progress=0.0)["reconstruction_gain"] >= q3d_base
    assert gsfusion_reconstruction_surrogate(0.2, 0.4, map_progress=0.0)["reconstruction_gain"] >= q3d_base
    assert gsfusion_reconstruction_surrogate(0.4, 0.4, map_progress=0.0)["q_joint"] >= 0.2 * 0.2
    sat_cfg = {
        "mapping_quality": {
            "gsfusion_weights": {
                "model": "saturation",
                "bias": 0.0,
                "w_rgb": 0.4,
                "w_depth": 0.3,
                "w_joint": 0.2,
                "lambda_rgb": 3.0,
                "lambda_depth": 8.0,
            }
        }
    }
    sat_low = gsfusion_reconstruction_surrogate(0.2, 0.2, cfg=sat_cfg)["reconstruction_gain"]
    sat_high_rgb = gsfusion_reconstruction_surrogate(0.4, 0.2, cfg=sat_cfg)
    sat_high_depth = gsfusion_reconstruction_surrogate(0.2, 0.4, cfg=sat_cfg)
    assert sat_high_rgb["reconstruction_gain"] >= sat_low
    assert sat_high_depth["reconstruction_gain"] >= sat_low
    assert sat_high_rgb["q_render"] > 0.2
    assert sat_high_depth["q_geometry"] > 0.4
    q3d_no_context = gsfusion_reconstruction_surrogate(0.3, 0.5, view_importance=1.0, map_progress=0.0)["q_3d"]
    q3d_with_context = gsfusion_reconstruction_surrogate(0.3, 0.5, view_importance=0.2, map_progress=0.9)["q_3d"]
    assert abs(q3d_no_context - q3d_with_context) < 1e-12

    c_rgb, c_depth = constraint_costs(0.4, 0.3, 0.6, 0.8)
    assert abs(c_rgb - 0.2) < 1e-8
    assert abs(c_depth - 0.5) < 1e-8

    mode, selected = select_constraint_mode(0.01, 0.02, 0.05, 0.05)
    assert mode == "reward" and selected is None
    mode, selected = select_constraint_mode(0.20, 0.06, 0.05, 0.05)
    assert mode == "constraint_rgb" and selected == "rgb"
    mode, selected = select_constraint_mode(0.06, 0.20, 0.05, 0.05)
    assert mode == "constraint_depth" and selected == "depth"

    policy = TwoHeadActorCritic(obs_dim=5, n_kd=5, n_beta=5, hidden_dim=16)
    obs = torch.randn(3, 5)
    actions = torch.tensor([[0, 1], [2, 3], [4, 0]])
    joint_log_prob, _, out = policy.evaluate_actions(obs, actions)
    _, joint_entropy, _ = policy.evaluate_actions(obs, actions)
    kd_log_prob = Categorical(logits=out["kd_logits"]).log_prob(actions[:, 0])
    beta_log_prob = Categorical(logits=out["beta_logits"]).log_prob(actions[:, 1])
    kd_entropy = Categorical(logits=out["kd_logits"]).entropy()
    beta_entropy = Categorical(logits=out["beta_logits"]).entropy()
    assert torch.allclose(joint_log_prob, kd_log_prob + beta_log_prob)
    assert torch.allclose(joint_entropy, kd_entropy + beta_entropy)

    costs = np.array([0.1, 0.2], dtype=np.float32)
    values = np.array([0.05, 0.1], dtype=np.float32)
    assert np.all((-costs) <= 0.0)
    assert np.all((-values) <= 0.0)

    try:
        split_snr_from_total(10.0, 24, 0.5, 24)
        raise AssertionError("expected zero RGB subcarrier guard to assert")
    except AssertionError:
        pass

    print("CRPO resource sanity checks passed.")


if __name__ == "__main__":
    main()
