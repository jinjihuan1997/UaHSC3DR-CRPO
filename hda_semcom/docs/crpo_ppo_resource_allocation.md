# CRPO-PPO Resource Allocation

This implementation follows `system_model.docx` for UAV-assisted disaster
building scanning with a fixed UAV trajectory and remote RGB-D reconstruction.
The RL agent does not control motion. It only allocates communication resources
between the RGB analog JSCC link and the digital depth link.

## State

The CMDP observation is:

```text
[reference_channel_quality, depth_payload_norm, map_progress, view_importance, task_progress]
```

The state intentionally does not contain action-dependent RGB or depth SNR.
Those SNRs are computed only after selecting `(k_d, beta_d)`.

## Action

The two-head categorical policy selects:

```text
k_d    = depth-link data subcarriers
beta_d = fraction of total transmit power assigned to depth
```

The remaining resources go to RGB:

```text
k_rgb = K_total - k_d
P_d   = beta_d P_total
P_rgb = (1 - beta_d) P_total
```

## A2G Channel And SNR Split

When a trajectory CSV is provided, the environment uses `src/channel/a2g.py` to
compute the A2G path loss from UAV and DRC positions. The CRPO environment then
converts path loss to linear link gain and computes two per-subcarrier SNRs:

```text
snr_rgb   = P_rgb G / (k_rgb B_sub N0 NF)
snr_depth = P_d   G / (k_d   B_sub N0 NF)
```

If no trajectory is provided, the offline table's reference SNR is converted
back into a reference link gain and the same split is applied.

## RGB Quality

RGB is transmitted through the analog JSCC semantic link. The environment first
tries to use the offline RGB PSNR lookup table and normalizes PSNR to `Q_rgb` in
`[0, 1]`. If an exact table entry is unavailable, it uses a monotonic JSCC proxy
of RGB SNR and RGB subcarrier count.

SNR is an intermediate variable only. It is not used as the final 3D metric.

## Depth Completeness

Depth is transmitted through the digital link. The environment computes an
802.11ah-like MCS bit budget from `snr_depth` and `k_d`, then converts it to a
partial depth-block success rate:

```text
R_depth = successful_depth_blocks / total_depth_blocks
```

`R_depth` is clipped to `[0, 1]`.

## GSFusion Surrogate

Real GSFusion reconstruction is not executed inside PPO rollouts. The current
training signal is a replaceable surrogate in `src/rl/mapping_quality.py`:

```text
q_3d = clip(bias
            + w_rgb Q_rgb
            + w_depth R_depth
            + w_joint Q_rgb R_depth,
            0, 1)
```

Default weights are:

```text
bias    = 0.00
w_rgb   = 0.35
w_depth = 0.40
w_joint = 0.25
```

This form reflects the RGB-D fusion logic in `system_model.docx`: RGB quality
supports appearance/rendering, depth completeness supports geometry, and the
joint term rewards frames where both modalities are usable.

The surrogate fitting features are only `Q_rgb`, `R_depth`, and
`Q_rgb * R_depth`. Resource variables such as `k_d`, `beta_d`, transmit power,
and `tx_amount_norm` are not direct fitting features; they affect `q_3d` only
through the communication chain that produces `Q_rgb` and `R_depth`.

The CRPO environment uses the immediate reward:

```text
reward = q_3d
```

`map_progress` remains a state variable tracking accumulated per-frame
reconstruction progress, but it is not a direct feature in the fitted
condition-level GSFusion surrogate.

Future real GSFusion metrics should replace this function behind the same
interface, either by running the backend or by using a learned/lookup proxy.

## Constraints

The CMDP tracks two instantaneous service-quality violation costs:

```text
c_R[t] = max(Q_req - Q_rgb[t], 0)
c_D[t] = max(D_req - R_depth[t], 0)
```

These are soft costs, not hard per-slot constraints. CRPO compares rollout
episode-average estimates with allowed violation levels:

```text
J_C_R <= epsilon_R
J_C_D <= epsilon_D
```

## CRPO-PPO Switching

This is a PPO-compatible engineering variant inspired by CRPO, not the original
NPG-style CRPO algorithm and not a claim of original CRPO convergence.

Per PPO update:

```text
if J_C_R <= epsilon_R and J_C_D <= epsilon_D:
    optimize reconstruction reward
else:
    select the most severely violated constraint:
        argmax_i (J_C_i - epsilon_i) / (epsilon_i + eta)
    optimize temporary reward = -selected_cost
```

The two-head policy log probability is the joint log probability:

```text
log pi(a) = log pi(k_d) + log pi(beta_d)
```

The PPO ratio uses that joint log probability. Cost critics estimate positive
expected costs, while constraint-mode actor advantages are computed from
negative costs.

## PPO-Penalty Baseline

The implementation also supports a standard PPO-penalty baseline:

```text
r_pen[t] = q_3d[t] - lambda_R c_R[t] - lambda_D c_D[t]
```

Unlike CRPO-PPO, PPO-penalty always optimizes this single scalar penalized
reward. It is simpler, but it is sensitive to the fixed penalty weights.

## Current Limitations

- `q_3d` is a simulation surrogate, not a real GSFusion evaluation.
- Offline lookup tables must be regenerated with `depth_payload_bits`; old
  `aux_payload_bits` CRPO tables are no longer accepted.
- The RGB lookup table has no `beta_d` dimension. `beta_d` affects RGB through
  the computed RGB SNR before table lookup.
- Evaluation should report `J_C_R` and `J_C_D` as the key constraint metrics,
  together with violation rates, average `Q_rgb`, average `R_depth`, average
  `k_d`, and average `beta_d`.
