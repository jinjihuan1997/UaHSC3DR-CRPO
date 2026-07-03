"""
简化版信道模型。

模拟路：AWGN，对复符号加循环对称复高斯噪声。
数字路：SNR -> MCS -> bit budget -> ROI token 预算截断 + 均匀量化。
        这里先实现可训练的预算近似，不包含真实 LDPC 编解码和调制解调。
"""
import math
import torch


def awgn(x_complex, snr_db):
    """对复符号加 AWGN。x_complex: (B, L) complex；snr_db: float 或 (B,) tensor。"""
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(float(snr_db), device=x_complex.device)
    noise_std = analog_noise_std(snr_db)
    if noise_std.dim() == 0:
        noise_std = noise_std.view(1, 1)
    else:
        noise_std = noise_std.view(-1, 1)
    n = torch.randn_like(x_complex.real) + 1j * torch.randn_like(x_complex.imag)
    return x_complex + noise_std * n


def analog_noise_std(snr_db):
    """单位平均功率复符号在给定 SNR 下的单实维 AWGN 标准差。"""
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(float(snr_db))
    snr = 10.0 ** (snr_db / 10.0)
    return (1.0 / snr).sqrt() / math.sqrt(2.0)


def default_mcs_table():
    """IEEE 802.11ah-like 1 MHz Long-GI LDPC-coded MCS abstraction."""
    return [
        {"mcs_index": 10, "snr_threshold": -1.0, "modulation": "BPSK_REP2",
         "n_bpscs": 1, "code_rate": 0.25, "ldpc_code_rate": 0.5,
         "repetition": 2, "n_dbps": 6, "phy_rate_mbps": 0.150},
        {"mcs_index": 0, "snr_threshold": 1.0, "modulation": "BPSK",
         "n_bpscs": 1, "code_rate": 0.5, "ldpc_code_rate": 0.5,
         "repetition": 1, "n_dbps": 12, "phy_rate_mbps": 0.300},
        {"mcs_index": 1, "snr_threshold": 4.0, "modulation": "QPSK",
         "n_bpscs": 2, "code_rate": 0.5, "ldpc_code_rate": 0.5,
         "repetition": 1, "n_dbps": 24, "phy_rate_mbps": 0.600},
        {"mcs_index": 2, "snr_threshold": 8.0, "modulation": "QPSK",
         "n_bpscs": 2, "code_rate": 0.75, "ldpc_code_rate": 0.75,
         "repetition": 1, "n_dbps": 36, "phy_rate_mbps": 0.900},
        {"mcs_index": 3, "snr_threshold": 10.0, "modulation": "16QAM",
         "n_bpscs": 4, "code_rate": 0.5, "ldpc_code_rate": 0.5,
         "repetition": 1, "n_dbps": 48, "phy_rate_mbps": 1.200},
        {"mcs_index": 4, "snr_threshold": 14.0, "modulation": "16QAM",
         "n_bpscs": 4, "code_rate": 0.75, "ldpc_code_rate": 0.75,
         "repetition": 1, "n_dbps": 72, "phy_rate_mbps": 1.800},
        {"mcs_index": 5, "snr_threshold": 17.0, "modulation": "64QAM",
         "n_bpscs": 6, "code_rate": 2.0 / 3.0, "ldpc_code_rate": 2.0 / 3.0,
         "repetition": 1, "n_dbps": 96, "phy_rate_mbps": 2.400},
        {"mcs_index": 6, "snr_threshold": 19.0, "modulation": "64QAM",
         "n_bpscs": 6, "code_rate": 0.75, "ldpc_code_rate": 0.75,
         "repetition": 1, "n_dbps": 108, "phy_rate_mbps": 2.700},
        {"mcs_index": 7, "snr_threshold": 24.0, "modulation": "64QAM",
         "n_bpscs": 6, "code_rate": 5.0 / 6.0, "ldpc_code_rate": 5.0 / 6.0,
         "repetition": 1, "n_dbps": 120, "phy_rate_mbps": 3.000},
        {"mcs_index": 8, "snr_threshold": 28.0, "modulation": "256QAM",
         "n_bpscs": 8, "code_rate": 0.75, "ldpc_code_rate": 0.75,
         "repetition": 1, "n_dbps": 144, "phy_rate_mbps": 3.600},
        {"mcs_index": 9, "snr_threshold": 30.0, "modulation": "256QAM",
         "n_bpscs": 8, "code_rate": 5.0 / 6.0, "ldpc_code_rate": 5.0 / 6.0,
         "repetition": 1, "n_dbps": 160, "phy_rate_mbps": 4.000},
    ]


def _ofdm_total_resource_elements(cfg):
    ch = cfg["channel"]
    slot_duration_s = float(ch.get("slot_duration_s", 0.02))
    t_sym_s = float(ch.get("ofdm_symbol_duration_us", 40.0)) * 1e-6
    data_subcarriers = int(ch.get("data_subcarriers", 24))
    n_ofdm_symbols = int(math.floor(slot_duration_s / t_sym_s))
    resource_fraction = float(ch.get("communication_resource_fraction", 1.0))
    resource_fraction = min(max(resource_fraction, 0.0), 1.0)
    total_re = max(1, int(math.floor(n_ofdm_symbols * data_subcarriers * resource_fraction)))
    return n_ofdm_symbols, total_re


def _adaptive_digital_ratio(snr, cfg):
    ch = cfg["channel"]
    lo = float(ch.get("adaptive_snr_min", -1.0))
    hi = float(ch.get("adaptive_snr_max", 30.0))
    r_lo = float(ch.get("digital_resource_ratio_min", 0.5))
    r_hi = float(ch.get("digital_resource_ratio_max", 0.85))
    if not torch.is_tensor(snr):
        snr = torch.tensor([float(snr)])
    t = ((snr.float() - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    return r_lo + (r_hi - r_lo) * t


def shared_ofdm_resource_budget(cfg, snr_db=None, for_model_init=False):
    """计算共享 1 MHz OFDM slot 中数字/模拟两路的 data RE 分配。"""
    ch = cfg["channel"]
    n_ofdm_symbols, total_re = _ofdm_total_resource_elements(cfg)
    data_subcarriers = int(ch.get("data_subcarriers", 24))
    resource_fraction = float(ch.get("communication_resource_fraction", 1.0))
    resource_fraction = min(max(resource_fraction, 0.0), 1.0)
    digital_subcarriers = None
    analog_subcarriers = None

    if ch.get("hda_resource_mode", "") == "shared_ofdm":
        allocation = ch.get("resource_allocation", "fixed")
        if allocation in ("fixed_subcarriers", "discrete_subcarriers"):
            digital_ratio = float(ch.get("digital_resource_ratio", 0.7))
            digital_ratio = min(max(digital_ratio, 0.0), 1.0)
            raw_subcarriers = data_subcarriers * digital_ratio
            digital_subcarriers = int(round(raw_subcarriers))
            if abs(raw_subcarriers - digital_subcarriers) > 1e-6:
                raise ValueError(
                    "fixed_subcarriers requires digital_resource_ratio = k / "
                    f"data_subcarriers; got ratio={digital_ratio} with "
                    f"data_subcarriers={data_subcarriers}"
                )
            digital_subcarriers = min(max(digital_subcarriers, 0), data_subcarriers)
            analog_subcarriers = data_subcarriers - digital_subcarriers
            digital_re = int(math.floor(n_ofdm_symbols * digital_subcarriers * resource_fraction))
            analog_symbols = max(1, int(math.floor(n_ofdm_symbols * analog_subcarriers * resource_fraction)))
        else:
            if allocation == "adaptive_snr":
                if for_model_init or snr_db is None:
                    digital_ratio = float(ch.get("digital_resource_ratio_min", 0.5))
                else:
                    digital_ratio = _adaptive_digital_ratio(snr_db, cfg)
            else:
                digital_ratio = float(ch.get("digital_resource_ratio", 0.7))
                digital_ratio = min(max(digital_ratio, 0.0), 1.0)

            if torch.is_tensor(digital_ratio):
                digital_re = torch.floor(digital_ratio * total_re).long()
                analog_symbols = (total_re - digital_re).clamp_min(1)
            else:
                digital_re = int(math.floor(total_re * digital_ratio))
                analog_symbols = max(1, total_re - digital_re)
    else:
        digital_re = total_re
        analog_symbols = int(cfg.get("model", {}).get("analog_symbols", 512))

    return {
        "n_ofdm_symbols": n_ofdm_symbols,
        "total_resource_elements": total_re,
        "digital_resource_elements": digital_re,
        "analog_symbols": analog_symbols,
        "digital_data_subcarriers": digital_subcarriers,
        "analog_data_subcarriers": analog_subcarriers,
        "effective_digital_resource_ratio": (None if digital_subcarriers is None
                                             else digital_subcarriers / max(data_subcarriers, 1)),
        "effective_analog_resource_ratio": (None if analog_subcarriers is None
                                            else analog_subcarriers / max(data_subcarriers, 1)),
    }


def mcs_lookup(snr_db, mcs_table=None, device=None):
    """按 SNR 查表，返回 spectral efficiency 和 MCS index。"""
    budget = digital_link_budget_from_snr(
        snr_db,
        {"channel": {
            "mcs_table": default_mcs_table() if mcs_table is None else mcs_table,
            "channel_bandwidth_hz": 1_000_000,
            "slot_duration_s": 1.0,
            "mac_efficiency": 1.0,
        }},
        device=device,
    )
    return budget["spectral_efficiency"], budget["mcs_index"]


def digital_link_budget_from_snr(snr_db, cfg, device=None):
    """返回 IEEE 802.11ah-like 数字上行链路预算。"""
    ch = cfg["channel"]
    mcs_table = ch.get("mcs_table", default_mcs_table())
    if not torch.is_tensor(snr_db):
        snr = torch.tensor([float(snr_db)], device=device)
    else:
        snr = snr_db.to(device=device).float().view(-1)

    se = torch.zeros_like(snr)
    idx = torch.full_like(snr, -1, dtype=torch.long)
    phy_rate_mbps = torch.zeros_like(snr)
    n_dbps = torch.zeros_like(snr)
    n_bpscs = torch.zeros_like(snr)
    code_rate = torch.zeros_like(snr)

    bandwidth_hz = float(ch.get("channel_bandwidth_hz", 2_000_000))
    slot_duration_s = float(ch.get("slot_duration_s", 0.02))
    mac_efficiency = float(ch.get("mac_efficiency", 0.8))
    t_sym_s = float(ch.get("ofdm_symbol_duration_us", 40.0)) * 1e-6
    resources = shared_ofdm_resource_budget(cfg, snr_db=snr, for_model_init=False)
    digital_re = resources["digital_resource_elements"]
    if not torch.is_tensor(digital_re):
        digital_re = torch.full_like(snr, float(digital_re))
    else:
        digital_re = digital_re.to(device=snr.device, dtype=snr.dtype).view(-1)

    for row in sorted(mcs_table, key=lambda x: float(x["snr_threshold"])):
        m = snr >= float(row["snr_threshold"])
        if "phy_rate_mbps" in row:
            rate_mbps = float(row["phy_rate_mbps"])
        else:
            rate_mbps = float(row["n_dbps"]) / t_sym_s / 1e6
        row_se = (rate_mbps * 1e6) / bandwidth_hz
        se = torch.where(m, torch.full_like(se, row_se), se)
        idx = torch.where(m, torch.full_like(idx, int(row["mcs_index"])), idx)
        phy_rate_mbps = torch.where(m, torch.full_like(phy_rate_mbps, rate_mbps), phy_rate_mbps)
        n_dbps = torch.where(m, torch.full_like(n_dbps, float(row["n_dbps"])), n_dbps)
        n_bpscs = torch.where(m, torch.full_like(n_bpscs, float(row["n_bpscs"])), n_bpscs)
        code_rate = torch.where(m, torch.full_like(code_rate, float(row["code_rate"])), code_rate)

    if ch.get("hda_resource_mode", "") == "shared_ofdm":
        budget_bits = digital_re * n_bpscs * code_rate * mac_efficiency
    else:
        budget_bits = phy_rate_mbps * 1e6 * slot_duration_s * mac_efficiency
    return {
        "mcs_index": idx,
        "spectral_efficiency": se,
        "phy_rate_mbps": phy_rate_mbps,
        "digital_budget_bits": budget_bits,
        "n_dbps": n_dbps,
        "n_bpscs": n_bpscs,
        "code_rate": code_rate,
        "total_resource_elements": resources["total_resource_elements"],
        "digital_resource_elements": digital_re.detach(),
        "analog_symbols": (resources["analog_symbols"].to(device=snr.device).view(-1)
                           if torch.is_tensor(resources["analog_symbols"])
                           else torch.full_like(snr, float(resources["analog_symbols"]))),
        "digital_data_subcarriers": resources.get("digital_data_subcarriers"),
        "analog_data_subcarriers": resources.get("analog_data_subcarriers"),
        "effective_digital_resource_ratio": resources.get("effective_digital_resource_ratio"),
        "effective_analog_resource_ratio": resources.get("effective_analog_resource_ratio"),
    }


def _ste_uniform_quantize(x, bits):
    """按样本动态范围做均匀量化，反向用 straight-through estimator。"""
    if bits <= 0:
        return torch.zeros_like(x)
    levels = float((1 << bits) - 1)
    clip = x.detach().abs().flatten(1).amax(dim=1).clamp_min(1e-6)
    view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
    clip = clip.view(*view_shape)
    x_clip = x.clamp(-clip, clip)
    x_norm = (x_clip + clip) / (2.0 * clip)
    q = torch.round(x_norm * levels) / levels
    x_q = q * (2.0 * clip) - clip
    return x + (x_q - x).detach()


def _token_budget_mask(mask_feat, budget_tokens):
    """按 ROI mask 分数保留每个样本 top-k 空间 token。"""
    B, _, H, W = mask_feat.shape
    scores = mask_feat.flatten(1)
    keep = torch.zeros_like(scores)
    total = H * W
    for b in range(B):
        k = int(budget_tokens[b].item())
        if k <= 0:
            continue
        k = min(k, total)
        idx = torch.topk(scores[b], k=k, largest=True).indices
        keep[b, idx] = 1.0
    return keep.view(B, 1, H, W)


def digital_channel(z_D, errorfree=True, bit_error_rate=0.0, training=True):
    """
    数字路简化模型。
    训练时：用加性均匀噪声替代量化，保持梯度可回传（论文做法）。
    errorfree=True：仅做量化噪声近似，无传输错误（数字路本应精确）。
    bit_error_rate>0：评估时模拟少量错误，对部分特征位置加扰动。
    """
    if training:
        # 量化误差 ~ U(-0.5, 0.5)，缩放到特征幅度范围
        u = torch.empty_like(z_D).uniform_(-0.5, 0.5) * 0.0  # 默认 0，可调
        z = z_D + u
    else:
        z = z_D
    if (not errorfree) and bit_error_rate > 0.0:
        mask = (torch.rand_like(z_D) < bit_error_rate).float()
        # 错误位置加随机扰动模拟解码失败
        z = z + mask * torch.randn_like(z_D) * z_D.abs().mean().clamp_min(1e-3)
    return z


def mcs_budgeted_digital_channel(z_D, mask_feat, snr_db, cfg, training=True):
    """
    第一版预算化数字路。

    每个空间 token 的代价近似为 C * quant_bits。SNR 通过 MCS 表换成每个
    channel use 可承载 bit 数，再决定保留多少 ROI token。
    """
    ch = cfg["channel"]
    quant_bits = int(ch.get("digital_quant_bits", 4))
    budget = digital_link_budget_from_snr(snr_db, cfg, z_D.device)
    se = budget["spectral_efficiency"]
    mcs_idx = budget["mcs_index"]
    budget_bits = budget["digital_budget_bits"]

    if budget_bits.numel() == 1 and z_D.shape[0] > 1:
        se = se.expand(z_D.shape[0])
        mcs_idx = mcs_idx.expand(z_D.shape[0])
        budget_bits = budget_bits.expand(z_D.shape[0])
        phy_rate_mbps = budget["phy_rate_mbps"].expand(z_D.shape[0])
        n_dbps = budget["n_dbps"].expand(z_D.shape[0])
        digital_re = budget["digital_resource_elements"].expand(z_D.shape[0])
        analog_symbols = budget["analog_symbols"].expand(z_D.shape[0])
    else:
        phy_rate_mbps = budget["phy_rate_mbps"]
        n_dbps = budget["n_dbps"]
        digital_re = budget["digital_resource_elements"]
        analog_symbols = budget["analog_symbols"]
    token_bits = max(1, z_D.shape[1] * quant_bits)
    budget_tokens = torch.floor(budget_bits / token_bits).long()

    keep_mask = _token_budget_mask(mask_feat, budget_tokens).to(z_D.dtype)
    z = _ste_uniform_quantize(z_D * keep_mask, quant_bits)

    if (not ch["train_digital_errorfree"]) and ch["digital_bit_error_rate"] > 0.0:
        z = digital_channel(
            z, errorfree=False,
            bit_error_rate=ch["digital_bit_error_rate"], training=training)

    info = {
        "mcs_index": mcs_idx.detach(),
        "spectral_efficiency": se.detach(),
        "digital_budget_bits": budget_bits.detach(),
        "digital_budget_tokens": budget_tokens.detach(),
        "phy_rate_mbps": phy_rate_mbps.detach(),
        "n_dbps": n_dbps.detach(),
        "digital_resource_elements": digital_re.detach(),
        "analog_symbols": analog_symbols.detach(),
        "digital_token_bits": token_bits,
        "keep_mask": keep_mask,  # 预算内的 ROI token 掩码，供模拟路补发预算外特征
    }
    return z, info
