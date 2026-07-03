"""Rebuild dataset manifests from scenes/*/{observations,damage_config_*}/traj*/frame_manifest.jsonl.

The generated records use full paths relative to dataset root, e.g.
    scenes/fushimi/observations/traj0/clean_images/000000.png
so the loader does not depend on the old damage_config directory name.
"""
import argparse
import json
import random
from pathlib import Path


def iter_trajectory_dirs(root, scene_filter=None):
    scenes_dir = root / "scenes"
    for scene_dir in sorted(scenes_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_filter and scene_dir.name != scene_filter:
            continue
        containers = []
        obs = scene_dir / "observations"
        if obs.is_dir():
            containers.append(obs)
        containers.extend(sorted(p for p in scene_dir.glob("damage_config_*") if p.is_dir()))
        for container in containers:
            for traj_dir in sorted(p for p in container.iterdir() if p.is_dir()):
                yield scene_dir.name, container.name, traj_dir


def frame_records(root, scene_id, container_name, traj_dir):
    manifest = traj_dir / "frame_manifest.jsonl"
    if manifest.exists():
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                yield normalize_record(root, scene_id, container_name, traj_dir, rec)
        return

    for image_path in sorted((traj_dir / "clean_images").glob("*.png")):
        stem = image_path.stem
        rec = {
            "scene_id": scene_id,
            "trajectory_id": traj_dir.name,
            "frame_id": int(stem) if stem.isdigit() else stem,
            "clean_image": f"clean_images/{image_path.name}",
            "depth": f"depth/{image_path.name}",
            "valid_mask": f"valid_masks/{image_path.name}",
            "camera_pose": f"poses/{stem}.json",
        }
        yield normalize_record(root, scene_id, container_name, traj_dir, rec)


def normalize_record(root, scene_id, container_name, traj_dir, rec):
    base_rel = Path("scenes") / scene_id / container_name / traj_dir.name
    out = dict(rec)
    out["scene_id"] = scene_id
    out["trajectory_id"] = traj_dir.name
    out["observation_group"] = container_name
    for key in ["clean_image", "roi_mask", "roi_priority_map", "depth", "valid_mask", "camera_pose"]:
        rel = out.get(key)
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.parts[:1] == ("scenes",):
            full_rel = rel_path
        else:
            full_rel = base_rel / rel_path
        if (root / full_rel).exists():
            out[key] = full_rel.as_posix()
        else:
            out[key] = full_rel.as_posix()
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset")
    ap.add_argument("--scene-id", default=None, help="Optional single scene to include, e.g. fushimi.")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    trajectories = []
    all_rows = []
    for scene_id, container_name, traj_dir in iter_trajectory_dirs(root, args.scene_id):
        rows = list(frame_records(root, scene_id, container_name, traj_dir))
        if rows:
            trajectories.append((scene_id, container_name, traj_dir.name, rows))
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(f"No samples found under {root}/scenes")

    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    n = len(trajectories)
    n_train = max(1, int(round(n * args.train_ratio))) if n > 1 else 1
    n_val = int(round(n * args.val_ratio)) if n - n_train > 1 else 0
    def flatten(items):
        return [rec for _, _, _, rows in items for rec in rows]

    if n == 1:
        rows = list(trajectories[0][3])
        n_rows = len(rows)
        n_train_rows = max(1, int(round(n_rows * args.train_ratio)))
        n_val_rows = int(round(n_rows * args.val_ratio)) if n_rows - n_train_rows > 1 else 0
        train_rows = rows[:n_train_rows]
        val_rows = rows[n_train_rows:n_train_rows + n_val_rows]
        test_rows = rows[n_train_rows + n_val_rows:]
        if not test_rows and len(rows) > 1:
            test_rows = train_rows[-1:]
            train_rows = train_rows[:-1]
        train_traj = trajectories
        val_traj = []
        test_traj = []
    else:
        train_traj = trajectories[:n_train]
        val_traj = trajectories[n_train:n_train + n_val]
        test_traj = trajectories[n_train + n_val:]
        if not test_traj:
            test_traj = val_traj[-1:] if val_traj else train_traj[-1:]
            if val_traj:
                val_traj = val_traj[:-1]
        train_rows = flatten(train_traj)
        val_rows = flatten(val_traj)
        test_rows = flatten(test_traj)

    manifest_dir = root / "manifests"
    write_jsonl(manifest_dir / "all_samples.jsonl", all_rows)
    write_jsonl(manifest_dir / "train_trajectory_split.jsonl", train_rows)
    write_jsonl(manifest_dir / "val_trajectory_split.jsonl", val_rows)
    write_jsonl(manifest_dir / "test_trajectory_split.jsonl", test_rows)
    write_jsonl(manifest_dir / "train_scene_split.jsonl", train_rows)
    write_jsonl(manifest_dir / "val_scene_split.jsonl", val_rows)
    write_jsonl(manifest_dir / "test_scene_split.jsonl", test_rows)

    print(f"scenes filter: {args.scene_id or 'all'}")
    print(f"trajectories: total={len(trajectories)} train={len(train_traj)} val={len(val_traj)} test={len(test_traj)}")
    print(f"samples: total={len(all_rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"wrote manifests to {manifest_dir}")


if __name__ == "__main__":
    main()
