"""Compare CRPO-PPO and PPO-penalty aggregate evaluation CSV files."""
import argparse
import csv
import os


def _to_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _scenario_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ("_aggregate", "-aggregate"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _method_rows(rows):
    return {row.get("method", ""): row for row in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aggregate_csv", nargs="+", help="Aggregate CSV files from eval_crpo_ppo.py.")
    ap.add_argument("--epsilon-rgb", type=float, required=True)
    ap.add_argument("--epsilon-depth", type=float, required=True)
    ap.add_argument("--out", default=None, help="Optional summary CSV path.")
    args = ap.parse_args()

    summary = []
    for path in args.aggregate_csv:
        rows = _read_rows(path)
        by_method = _method_rows(rows)
        crpo = by_method.get("CRPO-PPO")
        penalty = by_method.get("PPO-penalty")
        if crpo is None and penalty is None:
            continue

        for method, row in (("CRPO-PPO", crpo), ("PPO-penalty", penalty)):
            if row is None:
                continue
            j_c_r = _to_float(row, "J_C_R")
            j_c_d = _to_float(row, "J_C_D")
            excess_r = max(j_c_r - args.epsilon_rgb, 0.0)
            excess_d = max(j_c_d - args.epsilon_depth, 0.0)
            summary.append({
                "scenario": _scenario_name(path),
                "method": method,
                "avg_q3d": _to_float(row, "avg_q3d"),
                "avg_q_rgb": _to_float(row, "avg_q_rgb"),
                "avg_r_depth": _to_float(row, "avg_r_depth"),
                "J_C_R": j_c_r,
                "J_C_D": j_c_d,
                "excess_R": excess_r,
                "excess_D": excess_d,
                "total_excess": excess_r + excess_d,
                "feasible": int(excess_r <= 1e-12 and excess_d <= 1e-12),
                "rgb_violation_rate": _to_float(row, "rgb_violation_rate"),
                "depth_violation_rate": _to_float(row, "depth_violation_rate"),
                "avg_kd": _to_float(row, "avg_kd"),
                "avg_beta_d": _to_float(row, "avg_beta_d"),
                "episode_return": _to_float(row, "episode_return"),
            })

        if crpo is not None and penalty is not None:
            summary.append({
                "scenario": _scenario_name(path),
                "method": "CRPO-minus-PPO-penalty",
                "avg_q3d": _to_float(crpo, "avg_q3d") - _to_float(penalty, "avg_q3d"),
                "avg_q_rgb": _to_float(crpo, "avg_q_rgb") - _to_float(penalty, "avg_q_rgb"),
                "avg_r_depth": _to_float(crpo, "avg_r_depth") - _to_float(penalty, "avg_r_depth"),
                "J_C_R": _to_float(crpo, "J_C_R") - _to_float(penalty, "J_C_R"),
                "J_C_D": _to_float(crpo, "J_C_D") - _to_float(penalty, "J_C_D"),
                "excess_R": "",
                "excess_D": "",
                "total_excess": "",
                "feasible": "",
                "rgb_violation_rate": (
                    _to_float(crpo, "rgb_violation_rate")
                    - _to_float(penalty, "rgb_violation_rate")
                ),
                "depth_violation_rate": (
                    _to_float(crpo, "depth_violation_rate")
                    - _to_float(penalty, "depth_violation_rate")
                ),
                "avg_kd": _to_float(crpo, "avg_kd") - _to_float(penalty, "avg_kd"),
                "avg_beta_d": _to_float(crpo, "avg_beta_d") - _to_float(penalty, "avg_beta_d"),
                "episode_return": _to_float(crpo, "episode_return") - _to_float(penalty, "episode_return"),
            })

    if not summary:
        raise SystemExit("No CRPO-PPO or PPO-penalty rows found.")

    fields = list(summary[0].keys())
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

    print(",".join(fields))
    for row in summary:
        print(",".join(str(row.get(field, "")) for field in fields))


if __name__ == "__main__":
    main()
