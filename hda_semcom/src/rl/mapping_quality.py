"""GSFusion-style reconstruction-quality surrogate.

The real GSFusion reconstruction metric is not executed inside the RL loop.
This module keeps that boundary explicit: the current training signal is a
replaceable RGB-D surrogate, and future backends should be wired in here.
"""

import numpy as np


DEFAULT_GSFUSION_WEIGHTS = {
    "model": "linear",
    "bias": 0.0,
    "w_rgb": 0.35,
    "w_depth": 0.40,
    "w_joint": 0.25,
    "lambda_rgb": 1.0,
    "lambda_depth": 1.0,
}


def _mapping_cfg(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg.get("mapping_quality", cfg)
    return {}


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def gsfusion_reconstruction_surrogate(
    q_rgb,
    r_depth,
    *,
    view_importance=1.0,
    map_progress=0.0,
    cfg=None,
):
    """Return a reconstruction-aware RGB-D quality proxy.

    The surrogate follows the communication-to-reconstruction causal path:
    resource allocation affects SNR, SNR affects RGB quality and depth
    completeness, and only those received-observation qualities enter this
    condition-level reconstruction proxy.

        linear:
          q_3d = clip(bias + w_rgb Q_rgb + w_depth R_depth
                      + w_joint Q_rgb R_depth, 0, 1)

        saturation:
          g_rgb = 1 - exp(-lambda_rgb Q_rgb)
          g_depth = 1 - exp(-lambda_depth R_depth)
          q_3d = clip(bias + w_rgb g_rgb + w_depth g_depth
                      + w_joint g_rgb g_depth, 0, 1)

    view_importance and map_progress are accepted because the environment state
    still tracks them, but they are not direct fitting features here.
    """
    mapping_cfg = _mapping_cfg(cfg)
    mode = mapping_cfg.get("mapping_quality_mode", "gsfusion_surrogate")
    if mode != "gsfusion_surrogate":
        raise NotImplementedError(
            f"mapping_quality_mode={mode!r} is reserved for a future real or "
            "lookup-based reconstruction backend; only 'gsfusion_surrogate' "
            "is implemented in the RL loop."
        )

    weights = dict(DEFAULT_GSFUSION_WEIGHTS)
    weights.update(mapping_cfg.get("gsfusion_weights", {}))
    q_rgb = _clip01(q_rgb)
    r_depth = _clip01(r_depth)

    model = weights.get("model", "linear")
    if model == "linear":
        q_render = q_rgb
        q_geometry = r_depth
    elif model == "saturation":
        lambda_rgb = max(float(weights.get("lambda_rgb", 1.0)), 1e-12)
        lambda_depth = max(float(weights.get("lambda_depth", 1.0)), 1e-12)
        q_render = float(1.0 - np.exp(-lambda_rgb * q_rgb))
        q_geometry = float(1.0 - np.exp(-lambda_depth * r_depth))
    else:
        raise ValueError(f"unsupported gsfusion surrogate model: {model!r}")
    q_joint = q_rgb * r_depth
    if model == "saturation":
        q_joint = q_render * q_geometry
    q_3d = _clip01(
        float(weights.get("bias", 0.0))
        + float(weights["w_rgb"]) * q_render
        + float(weights["w_depth"]) * q_geometry
        + float(weights["w_joint"]) * q_joint
    )
    return {
        "q_render": float(q_render),
        "q_geometry": float(q_geometry),
        "q_joint": float(q_joint),
        "reconstruction_gain": float(q_3d),
        "q_3d": float(q_3d),
    }
