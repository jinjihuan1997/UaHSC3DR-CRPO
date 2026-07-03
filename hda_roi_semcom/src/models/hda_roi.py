"""
HDA-ROI-SemCom system assembly.

Current transmission mode:
    RGB image -> semantic JSCC / analog channel -> reconstructed RGB
    depth + valid mask -> lossless payload size estimate -> budgeted digital link

The ROI mask is no longer transmitted and no longer splits RGB features. It can
still be returned by the dataset for evaluation or for older experiments.
"""
import torch
import torch.nn as nn

from .semantic_codec import SemanticEncoder, SemanticDecoder
from .analog_channel import AnalogChannelEncoder, AnalogChannelDecoder
from ..channel.channels import awgn, digital_link_budget_from_snr, shared_ofdm_resource_budget


class HDAROISystem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        fd = m["feat_dim"]
        stride = m["feat_stride"]
        image_size = cfg["data"]["image_size"]
        if isinstance(image_size, (list, tuple)):
            img_h, img_w = int(image_size[0]), int(image_size[1])
        else:
            img_h = img_w = int(image_size)
        feat_hw = (img_h // stride, img_w // stride)
        self.feat_hw = feat_hw

        self.sem_enc = SemanticEncoder(3, fd, stride, m["swin_depth"],
                                       m["swin_heads"], m["swin_window"])
        self.sem_dec = SemanticDecoder(3, fd, stride, m["swin_depth"],
                                       m["swin_heads"], m["swin_window"])

        init_cfg = cfg
        ch = cfg["channel"]
        if ch.get("resource_allocation") == "discrete_subcarriers":
            data_subcarriers = int(ch.get("data_subcarriers", 24))
            grid = ch.get("digital_subcarrier_grid", [
                int(round(data_subcarriers * float(ch.get("digital_resource_ratio", 0.7))))
            ])
            min_digital = min(int(k) for k in grid)
            init_cfg = {**cfg, "channel": {**ch}}
            init_cfg["channel"]["resource_allocation"] = "fixed_subcarriers"
            init_cfg["channel"]["digital_resource_ratio"] = min_digital / float(data_subcarriers)

        self.analog_symbols = shared_ofdm_resource_budget(
            init_cfg, for_model_init=True)["analog_symbols"]
        self.ana_enc = AnalogChannelEncoder(fd, feat_hw, m["analog_hidden"],
                                            self.analog_symbols)
        self.ana_dec = AnalogChannelDecoder(fd, feat_hw, m["analog_hidden"],
                                            self.analog_symbols)

    def forward_codec(self, rgb):
        z = self.sem_enc(rgb)
        return self.sem_dec(z), z

    def _aux_digital_info(self, snr_db, aux_payload_bits, batch_size, device):
        budget = digital_link_budget_from_snr(snr_db, self.cfg, device=device)
        budget_bits = budget["digital_budget_bits"].to(device=device).float().view(-1)
        if budget_bits.numel() == 1 and batch_size > 1:
            budget_bits = budget_bits.expand(batch_size)

        if aux_payload_bits is None:
            payload_bits = torch.zeros(batch_size, device=device)
        else:
            payload_bits = aux_payload_bits.to(device=device).float().view(-1)
            if payload_bits.numel() == 1 and batch_size > 1:
                payload_bits = payload_bits.expand(batch_size)

        mcs = budget["mcs_index"].to(device=device).view(-1)
        phy_rate = budget["phy_rate_mbps"].to(device=device).view(-1)
        digital_re = budget["digital_resource_elements"].to(device=device).view(-1)
        if mcs.numel() == 1 and batch_size > 1:
            mcs = mcs.expand(batch_size)
            phy_rate = phy_rate.expand(batch_size)
            digital_re = digital_re.expand(batch_size)

        success = budget_bits >= payload_bits
        return {
            "mcs_index": mcs.detach(),
            "phy_rate_mbps": phy_rate.detach(),
            "aux_budget_bits": budget_bits.detach(),
            "aux_payload_bits": payload_bits.detach(),
            "aux_success": success.detach(),
            "digital_resource_elements": digital_re.detach(),
            "analog_symbols": budget["analog_symbols"].to(device=device).view(-1).detach(),
        }

    def forward(self, rgb, depth=None, valid_mask=None, snr_db=10.0,
                training=True, aux_payload_bits=None):
        z = self.sem_enc(rgb)
        x = self.ana_enc(z)

        resources = shared_ofdm_resource_budget(self.cfg, snr_db=snr_db, for_model_init=False)
        analog_counts = resources["analog_symbols"]
        if not torch.is_tensor(analog_counts):
            analog_counts = torch.tensor([float(analog_counts)], device=x.device)
        else:
            analog_counts = analog_counts.to(device=x.device).float().view(-1)
        if analog_counts.numel() == 1 and x.shape[0] > 1:
            analog_counts = analog_counts.expand(x.shape[0])

        sym_idx = torch.arange(x.shape[1], device=x.device).view(1, -1)
        analog_active = (sym_idx < analog_counts.view(-1, 1)).to(x.real.dtype)
        x = x * analog_active
        x_rx = awgn(x, snr_db) * analog_active
        z_hat = self.ana_dec(x_rx)
        rgb_hat = self.sem_dec(z_hat)

        aux_info = self._aux_digital_info(snr_db, aux_payload_bits, rgb.shape[0], rgb.device)
        aux_success = aux_info["aux_success"].to(device=rgb.device).view(-1, 1, 1, 1).to(rgb.dtype)
        depth_hat = depth * aux_success if depth is not None else None
        valid_mask_hat = valid_mask * aux_success if valid_mask is not None else None

        return {
            "rgb_hat": rgb_hat,
            "depth_hat": depth_hat,
            "valid_mask_hat": valid_mask_hat,
            "I_hat": rgb_hat,
            "z": z,
            "z_hat": z_hat,
            "aux_info": aux_info,
        }

    def codec_params(self):
        return list(self.sem_enc.parameters()) + list(self.sem_dec.parameters())

    def transceiver_params(self):
        return list(self.ana_enc.parameters()) + list(self.ana_dec.parameters())
