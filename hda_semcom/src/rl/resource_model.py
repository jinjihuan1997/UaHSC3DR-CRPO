"""Resource-allocation helpers for CRPO-inspired constrained PPO.

The CRPO logic here is an engineering variant used with PPO clipping. It
switches the PPO actor objective between mapping reward and the most violated
constraint, but it does not implement the original NPG-style CRPO update or
claim its convergence guarantees.
"""
import math

import numpy as np

from src.channel.channels import default_mcs_table


EPS = 1e-12


def action_to_resource(action, k_choices, beta_choices, k_total):
    """Map a two-head categorical action to physical resource values."""
    kd_idx, beta_idx = int(action[0]), int(action[1])
    k_d = int(k_choices[kd_idx])
    beta_d = float(beta_choices[beta_idx])
    k_rgb = int(k_total) - k_d
    assert k_d > 0
    assert k_rgb > 0
    assert 0.0 < beta_d < 1.0
    return {
        "kd_idx": kd_idx,
        "beta_idx": beta_idx,
        "k_d": k_d,
        "k_rgb": k_rgb,
        "beta_d": beta_d,
    }


def db_to_linear(db_value):
    return 10.0 ** (float(db_value) / 10.0)


def linear_to_db(linear_value):
    return 10.0 * math.log10(max(float(linear_value), EPS))


def dbm_to_watt(dbm):
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def watt_to_dbm(watt):
    return 10.0 * math.log10(max(float(watt), EPS)) + 30.0


def link_gain_from_path_loss(path_loss_db):
    return db_to_linear(-float(path_loss_db))


def power_split(beta_d, p_total_watt):
    assert 0.0 < float(beta_d) < 1.0
    assert float(p_total_watt) > 0.0
    p_d = float(beta_d) * float(p_total_watt)
    p_rgb = (1.0 - float(beta_d)) * float(p_total_watt)
    return p_d, p_rgb


def split_snr_from_link_gain(
    link_gain,
    k_d,
    beta_d,
    k_total,
    *,
    p_total_watt,
    bandwidth_hz,
    n0_watt_per_hz=None,
    noise_figure_db=None,
    noise_figure_linear=None,
):
    """Compute per-subcarrier RGB and depth-link SNR after allocation."""
    k_d = int(k_d)
    k_total = int(k_total)
    k_rgb = k_total - k_d
    p_d, p_rgb = power_split(beta_d, p_total_watt)
    b_sub_hz = float(bandwidth_hz) / max(k_total, 1)
    n0 = dbm_to_watt(-174.0) if n0_watt_per_hz is None else float(n0_watt_per_hz)
    nf = db_to_linear(noise_figure_db) if noise_figure_linear is None else float(noise_figure_linear)
    assert k_d > 0
    assert k_rgb > 0
    assert b_sub_hz > 0.0
    assert n0 > 0.0
    assert nf > 0.0
    assert float(link_gain) > 0.0

    snr_rgb = p_rgb * float(link_gain) / (k_rgb * b_sub_hz * n0 * nf)
    snr_depth = p_d * float(link_gain) / (k_d * b_sub_hz * n0 * nf)
    return {
        "snr_rgb_linear": float(snr_rgb),
        "snr_depth_linear": float(snr_depth),
        "snr_rgb_db": linear_to_db(snr_rgb),
        "snr_depth_db": linear_to_db(snr_depth),
        "p_d": float(p_d),
        "p_rgb": float(p_rgb),
        "b_sub_hz": float(b_sub_hz),
        "noise_power_per_subcarrier_w": float(b_sub_hz * n0 * nf),
    }


def split_snr_from_total(total_snr_db, k_d, beta_d, k_total):
    """Split total-link SNR into per-subcarrier RGB and depth SNRs.

    Existing A2G code returns the SNR obtained with total transmit power over
    the full data bandwidth. With equal subcarrier bandwidths, per-link SNR
    scales by power fraction and by K_total / allocated_subcarriers.
    """
    k_d = int(k_d)
    k_total = int(k_total)
    k_rgb = k_total - k_d
    assert k_d > 0
    assert k_rgb > 0
    assert 0.0 < float(beta_d) < 1.0

    total_snr_linear = db_to_linear(total_snr_db)
    snr_rgb = total_snr_linear * (1.0 - float(beta_d)) * k_total / k_rgb
    snr_d = total_snr_linear * float(beta_d) * k_total / k_d
    return linear_to_db(snr_rgb), linear_to_db(snr_d)


def rgb_quality_proxy(snr_rgb_db, k_rgb, k_total, snr_mid_db=8.0, snr_scale_db=4.0):
    """Monotonic placeholder for RGB semantic JSCC quality in [0, 1]."""
    q_snr = 1.0 / (1.0 + math.exp(-(float(snr_rgb_db) - snr_mid_db) / max(snr_scale_db, EPS)))
    q_res = math.sqrt(max(float(k_rgb), 0.0) / max(float(k_total), 1.0))
    return float(np.clip(q_snr * q_res, 0.0, 1.0))


def psnr_to_quality(psnr, psnr_min, psnr_max):
    denom = max(float(psnr_max) - float(psnr_min), 1e-8)
    return float(np.clip((float(psnr) - float(psnr_min)) / denom, 0.0, 1.0))


def highest_mcs_supported_by_snr(snr_db, mcs_table=None):
    selected = None
    for row in sorted(mcs_table or default_mcs_table(), key=lambda x: float(x["snr_threshold"])):
        if float(snr_db) >= float(row["snr_threshold"]):
            selected = row
    return selected


def depth_success_rate(
    snr_depth_db,
    k_d,
    depth_payload_bits,
    *,
    n_depth_blocks=16,
    slot_duration_s=0.5,
    ofdm_symbol_duration_us=40.0,
    communication_resource_fraction=0.8,
    mac_efficiency=0.8,
    mcs_table=None,
):
    """Compute partial depth completeness using MCS bit budget and blocks."""
    k_d = int(k_d)
    assert k_d > 0
    n_depth_blocks = max(int(n_depth_blocks), 1)
    n_symbols = int(math.floor(float(slot_duration_s) / (float(ofdm_symbol_duration_us) * 1e-6)))
    n_re_d = k_d * n_symbols * float(communication_resource_fraction)
    mcs = highest_mcs_supported_by_snr(snr_depth_db, mcs_table=mcs_table)
    if mcs is None:
        budget_bits = 0.0
        mcs_index = -1
        n_bpscs = 0.0
        code_rate = 0.0
    else:
        n_bpscs = float(mcs["n_bpscs"])
        code_rate = float(mcs["code_rate"])
        budget_bits = n_re_d * n_bpscs * code_rate * float(mac_efficiency)
        mcs_index = int(mcs["mcs_index"])
    payload_bits = max(float(depth_payload_bits), 0.0)
    if payload_bits <= EPS:
        depth_blocks_success = n_depth_blocks
    else:
        block_bits = payload_bits / n_depth_blocks
        depth_blocks_success = min(n_depth_blocks, int(math.floor(budget_bits / max(block_bits, EPS))))
    r_depth = depth_blocks_success / n_depth_blocks
    return {
        "r_depth": float(np.clip(r_depth, 0.0, 1.0)),
        "depth_bit_budget": float(budget_bits),
        "depth_payload_bits": float(payload_bits),
        "depth_blocks_success": int(depth_blocks_success),
        "depth_blocks_total": int(n_depth_blocks),
        "digital_re": float(n_re_d),
        "mcs": mcs_index,
        "n_bpscs": float(n_bpscs),
        "code_rate": float(code_rate),
    }


def constraint_costs(q_rgb, r_depth, q_req, depth_req):
    c_rgb = max(float(q_req) - float(q_rgb), 0.0)
    c_depth = max(float(depth_req) - float(r_depth), 0.0)
    return c_rgb, c_depth


def select_constraint_mode(j_c_r, j_c_d, epsilon_r, epsilon_d, eta=EPS):
    """Return reward mode or the most violated episode-average constraint."""
    j_c_r = float(j_c_r)
    j_c_d = float(j_c_d)
    epsilon_r = float(epsilon_r)
    epsilon_d = float(epsilon_d)
    rgb_violated = j_c_r > epsilon_r
    depth_violated = j_c_d > epsilon_d
    if not rgb_violated and not depth_violated:
        return "reward", None

    rgb_score = (j_c_r - epsilon_r) / (epsilon_r + float(eta)) if rgb_violated else -float("inf")
    depth_score = (
        (j_c_d - epsilon_d) / (epsilon_d + float(eta))
        if depth_violated else -float("inf")
    )
    if rgb_score >= depth_score:
        return "constraint_rgb", "rgb"
    return "constraint_depth", "depth"
