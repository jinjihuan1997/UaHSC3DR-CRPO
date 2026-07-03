"""CMDP environment for CRPO-inspired UAV SemCom resource allocation.

The UAV trajectory is fixed or sampled externally. The action only allocates
data subcarriers and transmit-power fraction between RGB JSCC and the digital
depth link.
"""
import csv

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.channel.a2g import A2GChannel, load_trajectory_csv


def _load_per_trajectory_surrogate_yaml(path):
    """Return dict[traj_id -> gsfusion_weights dict] from per_trajectory_surrogates YAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for per-trajectory surrogate YAML; run: pip install pyyaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    result = {}
    for traj_id, entry in data.get("per_trajectory_surrogates", {}).items():
        result[str(traj_id)] = {k: entry[k] for k in
            ("model", "bias", "w_rgb", "w_depth", "w_joint", "lambda_rgb", "lambda_depth")
            if k in entry}
    return result
from src.channel.channels import default_mcs_table
from src.rl.mapping_quality import gsfusion_reconstruction_surrogate
from src.rl.resource_model import (
    action_to_resource,
    constraint_costs,
    db_to_linear,
    dbm_to_watt,
    depth_success_rate,
    link_gain_from_path_loss,
    psnr_to_quality,
    rgb_quality_proxy,
    split_snr_from_link_gain,
)


DEFAULT_K_D_CHOICES = [10, 12, 14, 16, 18, 20, 22]
DEFAULT_BETA_D_CHOICES = [0.2, 0.4, 0.6, 0.8]


class CMDPResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        table_csv,
        episode_len=100,
        seed=None,
        trajectory_csv=None,
        channel_kwargs=None,
        trajectory_reset="trajectory",
        trajectory_ids=None,
        k_total=24,
        k_d_choices=None,
        beta_d_choices=None,
        q_req=0.6,
        depth_req=0.8,
        map_progress_max=None,
        slot_duration_s=0.5,
        ofdm_symbol_duration_us=40.0,
        communication_resource_fraction=0.8,
        mac_efficiency=0.8,
        n_depth_blocks=16,
        mcs_table=None,
        snr_mid_db=8.0,
        snr_scale_db=4.0,
        bandwidth_hz=1e6,
        p_total_watt=None,
        ptx_dbm=20.0,
        noise_figure_db=7.0,
        mapping_quality_mode="gsfusion_surrogate",
        mapping_quality_cfg=None,
        per_trajectory_surrogate_yaml=None,
        psnr_ref_min=None,
        psnr_ref_max=None,
        payload_ref_min=None,
        payload_ref_max=None,
    ):
        super().__init__()
        self.table_csv = table_csv
        self.episode_len = int(episode_len)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.k_total = int(k_total)
        self.k_d_choices = [int(x) for x in (k_d_choices or DEFAULT_K_D_CHOICES)]
        self.beta_d_choices = [float(x) for x in (beta_d_choices or DEFAULT_BETA_D_CHOICES)]
        for k_d in self.k_d_choices:
            if k_d <= 0 or k_d >= self.k_total:
                raise ValueError(f"k_d choices must be in (0, {self.k_total}), got {k_d}")
        for beta in self.beta_d_choices:
            if beta <= 0.0 or beta >= 1.0:
                raise ValueError(f"beta_d choices must be in (0, 1), got {beta}")

        self.q_req = float(q_req)
        self.depth_req = float(depth_req)
        self.map_progress_max = float(map_progress_max if map_progress_max is not None else 1.0)
        self.slot_duration_s = float(slot_duration_s)
        self.ofdm_symbol_duration_us = float(ofdm_symbol_duration_us)
        self.communication_resource_fraction = float(communication_resource_fraction)
        self.mac_efficiency = float(mac_efficiency)
        self.n_depth_blocks = int(n_depth_blocks)
        self.mcs_table = mcs_table or default_mcs_table()
        self.snr_mid_db = float(snr_mid_db)
        self.snr_scale_db = float(snr_scale_db)
        self.bandwidth_hz = float(bandwidth_hz)
        if channel_kwargs:
            ptx_dbm = float(channel_kwargs.get("ptx_dbm", ptx_dbm))
            self.bandwidth_hz = float(channel_kwargs.get("bandwidth_hz", self.bandwidth_hz))
            noise_figure_db = float(channel_kwargs.get("noise_figure_db", noise_figure_db))
        self.p_total_watt = float(p_total_watt) if p_total_watt is not None else dbm_to_watt(ptx_dbm)
        self.noise_figure_db = float(noise_figure_db)
        self.noise_figure_linear = db_to_linear(self.noise_figure_db)
        self.n0_watt_per_hz = dbm_to_watt(-174.0)
        self.mapping_quality_cfg = dict(mapping_quality_cfg or {})
        self.mapping_quality_cfg.setdefault("mapping_quality_mode", mapping_quality_mode)
        self.per_trajectory_weights = (
            _load_per_trajectory_surrogate_yaml(per_trajectory_surrogate_yaml)
            if per_trajectory_surrogate_yaml else {}
        )
        self._psnr_ref_min = float(psnr_ref_min) if psnr_ref_min is not None else None
        self._psnr_ref_max = float(psnr_ref_max) if psnr_ref_max is not None else None
        self._payload_ref_min = float(payload_ref_min) if payload_ref_min is not None else None
        self._payload_ref_max = float(payload_ref_max) if payload_ref_max is not None else None

        if trajectory_reset not in {"random", "zero", "trajectory"}:
            raise ValueError("trajectory_reset must be 'random', 'zero', or 'trajectory'")
        self.trajectory_reset = trajectory_reset
        self.trajectory_rows = load_trajectory_csv(trajectory_csv) if trajectory_csv else None
        self.trajectory_ids = self._normalize_traj_ids(trajectory_ids)
        self.trajectory_groups = self._group_trajectories(self.trajectory_rows) if self.trajectory_rows else []
        self.trajectory_group = None
        self.trajectory_idx = 0
        self.channel = A2GChannel(**(channel_kwargs or {})) if trajectory_csv else None
        self.channel_info = None
        self._table_trajectory_groups = []  # filled after _load_table when no trajectory_csv

        self.rows = []
        self.by_key = {}
        self.by_traj_frame_key = {}
        self.sample_by_traj_step = {}
        self.sample_state_rows = []
        self.sample_rows = []
        self._load_table(table_csv)
        if self.trajectory_groups:
            self.trajectory_groups = self._filter_trajectory_groups(self.trajectory_groups)
            if not self.trajectory_groups:
                raise ValueError("trajectory_csv has no trajectory/frame entries present in the offline table")
        elif trajectory_reset == "trajectory":
            # No trajectory_csv: build groups from lookup table rows directly.
            self._table_trajectory_groups = self._build_table_trajectory_groups()

        self.table_k_list = sorted({int(r["k_d"]) for r in self.rows})
        self.snr_list = sorted({float(r["snr_db"]) for r in self.rows})
        # Use external reference normalization if provided (ensures train/test consistency).
        # Otherwise fall back to table min/max (original behavior).
        self.psnr_min = self._psnr_ref_min if self._psnr_ref_min is not None else min(float(r["rgb_psnr"]) for r in self.rows)
        self.psnr_max = self._psnr_ref_max if self._psnr_ref_max is not None else max(float(r["rgb_psnr"]) for r in self.rows)
        self.payload_min = self._payload_ref_min if self._payload_ref_min is not None else min(float(r["depth_payload_bits"]) for r in self.rows)
        self.payload_max = self._payload_ref_max if self._payload_ref_max is not None else max(float(r["depth_payload_bits"]) for r in self.rows)

        self.action_space = spaces.MultiDiscrete([len(self.k_d_choices), len(self.beta_d_choices)])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.current = None
        self.map_progress = 0.0
        self.map_progress_final = 0.0

    @staticmethod
    def _to_int(value, default=None):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _norm(value, lo, hi):
        denom = max(float(hi) - float(lo), 1e-8)
        return float(np.clip((float(value) - float(lo)) / denom, 0.0, 1.0))

    @staticmethod
    def _group_trajectories(rows):
        groups = {}
        for row in rows or []:
            key = row.get("trajectory_id", "")
            groups.setdefault(key, []).append(row)
        return [
            sorted(group, key=lambda row: CMDPResourceAllocationEnv._to_int(row.get("step"), 0))
            for group in groups.values()
        ]

    @staticmethod
    def _normalize_traj_id(value):
        if value in ("", None):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if text.startswith("traj"):
            return text
        try:
            return f"traj{int(float(text))}"
        except ValueError:
            return text

    @classmethod
    def _normalize_traj_ids(cls, values):
        if values is None:
            return None
        if isinstance(values, str):
            values = [x for x in values.split(",") if x.strip()]
        out = {cls._normalize_traj_id(value) for value in values}
        out.discard("")
        return out or None

    def _row_traj_id(self, row):
        return row.get("trajectory_id_norm") or self._normalize_traj_id(row.get("trajectory_id", ""))

    def _load_table(self, table_csv):
        with open(table_csv, newline="") as f:
            reader = csv.DictReader(f)
            required = {"sample_idx", "snr_db", "k_d", "rgb_psnr", "depth_payload_bits"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{table_csv} missing required columns: {sorted(missing)}")
            for raw in reader:
                row = dict(raw)
                row["sample_idx"] = int(float(row["sample_idx"]))
                row["snr_db"] = float(row["snr_db"])
                row["k_d"] = int(float(row["k_d"]))
                row["rgb_psnr"] = float(row["rgb_psnr"])
                row["depth_payload_bits"] = float(row["depth_payload_bits"])
                row["trajectory_id_norm"] = self._normalize_traj_id(row.get("trajectory_id", ""))
                row["frame_id_int"] = self._to_int(row.get("frame_id"))
                if self.trajectory_ids is not None and row["trajectory_id_norm"] not in self.trajectory_ids:
                    continue
                key = (row["sample_idx"], row["snr_db"], row["k_d"])
                self.rows.append(row)
                self.by_key[key] = row
                if row["trajectory_id_norm"] and row["frame_id_int"] is not None:
                    traj_key = (row["trajectory_id_norm"], row["frame_id_int"], row["snr_db"], row["k_d"])
                    self.by_traj_frame_key[traj_key] = row
        if not self.rows:
            raise ValueError(f"empty offline lookup table: {table_csv}")

        state_map = {}
        sample_map = {}
        for row in self.rows:
            state_map[(row["sample_idx"], row["snr_db"])] = row
            sample_map[row["sample_idx"]] = row
            traj_id = row.get("trajectory_id_norm") or self._normalize_traj_id(row.get("trajectory_id", ""))
            frame_id = row.get("frame_id_int")
            if traj_id and frame_id is not None:
                self.sample_by_traj_step[(traj_id, frame_id)] = row
        self.sample_state_rows = list(state_map.values())
        self.sample_rows = list(sample_map.values())

    def _build_table_trajectory_groups(self):
        """Group lookup table rows by trajectory_id for trajectory-mode episodes without A2G CSV."""
        from collections import defaultdict
        groups = defaultdict(list)
        for row in self.rows:
            traj_id = row.get("trajectory_id_norm", "")
            if traj_id:
                groups[traj_id].append(row)
        result = [
            sorted(rows, key=lambda r: r.get("frame_id_int") or 0)
            for rows in groups.values()
        ]
        if self.trajectory_ids is not None:
            result = [g for g in result if
                      self._normalize_traj_id(g[0].get("trajectory_id_norm", "")) in self.trajectory_ids]
        return result

    def _apply_per_trajectory_surrogate(self):
        """Switch surrogate weights to match the current trajectory group, if available."""
        if not self.per_trajectory_weights or not self.trajectory_group:
            return
        traj_id = self._row_traj_id(self.trajectory_group[0])
        weights = self.per_trajectory_weights.get(traj_id)
        if weights:
            self.mapping_quality_cfg["gsfusion_weights"] = dict(weights)

    def _filter_trajectory_groups(self, groups):
        filtered = []
        for group in groups:
            traj_id = self._row_traj_id(group[0]) if group else ""
            if self.trajectory_ids is not None and traj_id not in self.trajectory_ids:
                continue
            if any(self._sample_for_trajectory_row(row) is not None for row in group):
                filtered.append(group)
        return filtered

    def _sample_for_trajectory_row(self, row):
        traj_id = self._row_traj_id(row)
        step = self._to_int(row.get("step"))
        if step is None:
            return None
        return self.sample_by_traj_step.get((traj_id, step))

    def _nearest_snr(self, snr_db):
        return min(self.snr_list, key=lambda x: abs(float(x) - float(snr_db)))

    def _nearest_table_k(self, k_d):
        return min(self.table_k_list, key=lambda x: abs(int(x) - int(k_d)))

    def _lookup_rgb_quality(self, sample_idx, snr_rgb_db, k_d, k_rgb):
        table_snr = self._nearest_snr(snr_rgb_db)
        frame_id = self.current.get("frame_id_int") if self.current is not None else None
        traj_id = self.current.get("trajectory_id_norm", "") if self.current is not None else ""
        key = (int(sample_idx), table_snr, int(k_d))
        row = self.by_key.get(key)
        if traj_id and frame_id is not None:
            traj_key = (traj_id, frame_id, table_snr, int(k_d))
            row = self.by_traj_frame_key.get(traj_key) or row
        lookup_used = row is not None
        if row is None:
            fallback_k = self._nearest_table_k(k_d)
            row = self.by_key.get((int(sample_idx), table_snr, fallback_k))
            if traj_id and frame_id is not None:
                row = self.by_traj_frame_key.get((traj_id, frame_id, table_snr, fallback_k)) or row
        if row is not None:
            return psnr_to_quality(row["rgb_psnr"], self.psnr_min, self.psnr_max), table_snr, lookup_used
        return (
            rgb_quality_proxy(
                snr_rgb_db,
                k_rgb,
                self.k_total,
                snr_mid_db=self.snr_mid_db,
                snr_scale_db=self.snr_scale_db,
            ),
            table_snr,
            False,
        )

    def _derive_link_gain_from_reference_snr(self, snr_db):
        total_snr_linear = db_to_linear(snr_db)
        noise_power = self.bandwidth_hz * self.n0_watt_per_hz * self.noise_figure_linear
        return total_snr_linear * noise_power / max(self.p_total_watt, 1e-12)

    def _set_channel_state(self, channel_info=None):
        if channel_info is not None:
            self.current["actual_snr_db"] = float(channel_info["snr_db"])
            if "link_gain" in channel_info:
                self.current["link_gain"] = float(channel_info["link_gain"])
            else:
                self.current["link_gain"] = link_gain_from_path_loss(channel_info["path_loss_db"])
        else:
            self.current["actual_snr_db"] = float(self.current["snr_db"])
            self.current["link_gain"] = self._derive_link_gain_from_reference_snr(self.current["actual_snr_db"])

    def _view_importance(self):
        for key in ("view_importance", "roi_ratio", "new_coverage_ratio"):
            if key in self.current and self.current[key] not in ("", None):
                try:
                    return float(np.clip(float(self.current[key]), 0.0, 1.0))
                except ValueError:
                    pass
        return 1.0

    def _obs(self):
        return np.array([
            self._norm(self.current["actual_snr_db"], min(self.snr_list), max(self.snr_list)),
            self._norm(self.current["depth_payload_bits"], self.payload_min, self.payload_max),
            float(np.clip(self.map_progress / max(self.map_progress_max, 1e-8), 0.0, 1.0)),
            self._view_importance(),
            float(np.clip(self.t / max(self.episode_len, 1), 0.0, 1.0)),
        ], dtype=np.float32)

    def _sample_state(self):
        if self.trajectory_reset == "trajectory" and self.trajectory_group is not None:
            while self.trajectory_idx < len(self.trajectory_group):
                trajectory_row = self.trajectory_group[self.trajectory_idx]
                self.trajectory_idx += 1
                if self.channel is not None:
                    sample_row = self._sample_for_trajectory_row(trajectory_row)
                    if sample_row is None:
                        continue
                    self.current = dict(sample_row)
                    self.channel_info = self.channel.snr_from_row(trajectory_row, self.rng)
                    self._set_channel_state(self.channel_info)
                else:
                    # Table-based trajectory mode: row IS the lookup row
                    self.current = dict(trajectory_row)
                    self.channel_info = None
                    self._set_channel_state(None)
                return True
            return False

        if self.channel is None:
            idx = int(self.rng.integers(0, len(self.sample_state_rows)))
            self.current = dict(self.sample_state_rows[idx])
            self.channel_info = None
            self._set_channel_state(None)
            return True

        # WARNING: random-reset mode with A2G channel draws the image frame randomly
        # but the channel row sequentially from the trajectory — the two are mismatched
        # (different spatial positions). This path is only reached when
        # trajectory_reset != "trajectory", which is not used in the main training pipeline.
        # If you need this path, refactor so that current and channel_info come from the
        # same lookup row.
        import warnings
        warnings.warn(
            "random-reset with A2G channel: image frame and channel state are drawn from "
            "different positions. Use trajectory_reset='trajectory' for spatially consistent samples.",
            UserWarning, stacklevel=4,
        )
        idx = int(self.rng.integers(0, len(self.sample_rows)))
        self.current = dict(self.sample_rows[idx])
        active_rows = self.trajectory_group if self.trajectory_group is not None else self.trajectory_rows
        self.channel_info = self.channel.snr_from_row(active_rows[self.trajectory_idx], self.rng)
        self._set_channel_state(self.channel_info)
        self.trajectory_idx = (self.trajectory_idx + 1) % len(active_rows)
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.map_progress = 0.0
        self.map_progress_final = 0.0
        if self.trajectory_rows:
            if self.trajectory_reset == "trajectory":
                group_idx = int(self.rng.integers(0, len(self.trajectory_groups)))
                self.trajectory_group = self.trajectory_groups[group_idx]
                self.trajectory_idx = 0
                self._apply_per_trajectory_surrogate()
            elif self.trajectory_reset == "zero":
                self.trajectory_group = None
                self.trajectory_idx = 0
            else:
                self.trajectory_group = None
                self.trajectory_idx = int(self.rng.integers(0, len(self.trajectory_rows)))
        elif self._table_trajectory_groups and self.trajectory_reset == "trajectory":
            group_idx = int(self.rng.integers(0, len(self._table_trajectory_groups)))
            self.trajectory_group = self._table_trajectory_groups[group_idx]
            self.trajectory_idx = 0
            self._apply_per_trajectory_surrogate()
        if not self._sample_state():
            raise RuntimeError("failed to sample an initial trajectory state")
        return self._obs(), {}

    def step(self, action):
        resource = action_to_resource(action, self.k_d_choices, self.beta_d_choices, self.k_total)
        snr_info = split_snr_from_link_gain(
            self.current["link_gain"],
            resource["k_d"],
            resource["beta_d"],
            self.k_total,
            p_total_watt=self.p_total_watt,
            bandwidth_hz=self.bandwidth_hz,
            n0_watt_per_hz=self.n0_watt_per_hz,
            noise_figure_linear=self.noise_figure_linear,
        )
        snr_rgb_db = snr_info["snr_rgb_db"]
        snr_depth_db = snr_info["snr_depth_db"]
        q_rgb, table_snr_rgb_db, rgb_lookup_exact = self._lookup_rgb_quality(
            self.current["sample_idx"], snr_rgb_db, resource["k_d"], resource["k_rgb"])
        depth_info = depth_success_rate(
            snr_depth_db,
            resource["k_d"],
            self.current["depth_payload_bits"],
            n_depth_blocks=self.n_depth_blocks,
            slot_duration_s=self.slot_duration_s,
            ofdm_symbol_duration_us=self.ofdm_symbol_duration_us,
            communication_resource_fraction=self.communication_resource_fraction,
            mac_efficiency=self.mac_efficiency,
            mcs_table=self.mcs_table,
        )
        r_depth = depth_info["r_depth"]
        view_importance = self._view_importance()
        quality_info = gsfusion_reconstruction_surrogate(
            q_rgb=q_rgb,
            r_depth=r_depth,
            view_importance=view_importance,
            map_progress=self.map_progress,
            cfg=self.mapping_quality_cfg,
        )
        q_3d = quality_info["q_3d"]
        reward = float(q_3d)
        c_rgb, c_depth = constraint_costs(q_rgb, r_depth, self.q_req, self.depth_req)
        map_increment = float(q_3d) / max(float(self.episode_len), 1.0)
        self.map_progress = float(np.clip(self.map_progress + map_increment, 0.0, self.map_progress_max))
        self.map_progress_final = float(self.map_progress)   # episode-cumulative avg q_3d at episode end

        self.t += 1
        trajectory_ended = (
            self.trajectory_rows is not None
            and self.trajectory_reset == "trajectory"
            and self.trajectory_idx >= len(self.trajectory_group)
        )
        terminated = self.t >= self.episode_len or trajectory_ended
        truncated = False
        info = {
            "sample_idx": int(self.current["sample_idx"]),
            "trajectory_id": self.current.get("trajectory_id_norm", self.current.get("trajectory_id", "")),
            "frame_id": int(self.current["frame_id_int"]) if self.current.get("frame_id_int") is not None else -1,
            "total_snr_db": float(self.current["actual_snr_db"]),
            "snr_rgb_db": float(snr_rgb_db),
            "snr_depth_db": float(snr_depth_db),
            "table_snr_rgb_db": float(table_snr_rgb_db),
            "rgb_lookup_exact": bool(rgb_lookup_exact),
            "k_d": int(resource["k_d"]),
            "k_rgb": int(resource["k_rgb"]),
            "beta_d": float(resource["beta_d"]),
            "p_d": float(snr_info["p_d"]),
            "p_rgb": float(snr_info["p_rgb"]),
            "link_gain": float(self.current["link_gain"]),
            "q_rgb": float(q_rgb),
            "r_depth": float(r_depth),
            "q_render": float(quality_info["q_render"]),
            "q_geometry": float(quality_info["q_geometry"]),
            "q_joint": float(quality_info["q_joint"]),
            "reconstruction_gain": float(quality_info["reconstruction_gain"]),
            "q_3d": float(q_3d),
            "cost_rgb": float(c_rgb),
            "cost_depth": float(c_depth),
            "reward": float(reward),
            "map_increment": float(map_increment),
            "map_progress": float(self.map_progress),
            "view_importance": float(view_importance),
            "depth_payload_bits": float(self.current["depth_payload_bits"]),
            "depth_bit_budget": float(depth_info["depth_bit_budget"]),
            "depth_blocks_success": int(depth_info["depth_blocks_success"]),
            "depth_blocks_total": int(depth_info["depth_blocks_total"]),
            "digital_re": float(depth_info["digital_re"]),
            "mcs_index": int(depth_info["mcs"]),
            "mcs": int(depth_info["mcs"]),
        }
        if self.channel_info is not None:
            info.update({
                "path_loss_db": self.channel_info["path_loss_db"],
                "shadow_db": self.channel_info["shadow_db"],
                "distance_m": self.channel_info["distance_m"],
                "ptx_dbm": self.channel_info["ptx_dbm"],
                "ptx_w": self.channel_info["ptx_w"],
            })

        if not terminated:
            if not self._sample_state():
                terminated = True
        return self._obs(), reward, terminated, truncated, info
