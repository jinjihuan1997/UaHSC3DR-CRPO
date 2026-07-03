"""Cache lightweight geometry samples from GSFusion meshes.

For predicted GSFusion meshes this samples mesh vertices uniformly.  The
GSFusion TSDF marching-cubes output is huge ASCII PLY, so vertex sampling is
the practical storage-relief path.  For a single GT OBJ/PLY, optional surface
sampling is supported when faces are available.
"""

import argparse
import json
import re
import shutil
import zlib
from pathlib import Path

import numpy as np


def _latest_mesh(mesh_dir):
    paths = sorted(Path(mesh_dir).glob("mesh_*.ply"))
    if not paths:
        return None

    def key(path):
        m = re.search(r"mesh_(\d+)\.ply", path.name)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=key)[-1]


def _sample_indices(n, k, seed):
    k = min(k, n)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    vals = set()
    while len(vals) < k:
        need = k - len(vals)
        vals.update(int(x) for x in rng.integers(0, n, size=max(need * 2, 1024)))
    arr = np.fromiter(vals, dtype=np.int64)
    if len(arr) > k:
        arr = rng.choice(arr, size=k, replace=False)
    return np.sort(arr)


def _parse_ply_header(path):
    with Path(path).open("rb") as f:
        header_lines = []
        header_bytes = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"bad ply header: {path}")
            header_bytes += len(line)
            text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(text)
            if text == "end_header":
                break

    fmt = next((line for line in header_lines if line.startswith("format ")), None)
    vertex_count = 0
    properties = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
            in_vertex = True
            continue
        if line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex = False
        if in_vertex and line.startswith("property "):
            parts = line.split()
            properties.append((parts[1], parts[2]))
    prop_names = [name for _, name in properties]
    xyz_idx = [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]
    return fmt, vertex_count, properties, xyz_idx, header_bytes


def _sample_ply_vertices(path, sample_points, seed):
    fmt, vertex_count, properties, xyz_idx, header_bytes = _parse_ply_header(path)
    if vertex_count <= 0:
        raise ValueError(f"no vertices in ply: {path}")

    indices = _sample_indices(vertex_count, sample_points, seed)
    if "ascii" in fmt:
        pts = np.empty((len(indices), 3), dtype=np.float32)
        target_pos = 0
        with Path(path).open("rb") as f:
            f.seek(header_bytes)
            for row_idx in range(vertex_count):
                line = f.readline()
                if target_pos >= len(indices):
                    break
                if row_idx != indices[target_pos]:
                    continue
                parts = line.split()
                pts[target_pos] = [
                    float(parts[xyz_idx[0]]),
                    float(parts[xyz_idx[1]]),
                    float(parts[xyz_idx[2]]),
                ]
                target_pos += 1
        if target_pos != len(indices):
            raise ValueError(f"truncated vertex data in ply: {path}")
        return pts, {"source_vertex_count": vertex_count, "method": "uniform_ply_vertex_sample"}

    if "binary_little_endian" in fmt:
        type_map = {
            "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
            "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
            "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        dtype = np.dtype([(name, type_map[t]) for t, name in properties])
        with Path(path).open("rb") as f:
            f.seek(header_bytes)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        prop_names = [name for _, name in properties]
        pts_all = np.column_stack([
            data[prop_names[xyz_idx[0]]],
            data[prop_names[xyz_idx[1]]],
            data[prop_names[xyz_idx[2]]],
        ]).astype(np.float32, copy=False)
        return pts_all[indices], {"source_vertex_count": vertex_count, "method": "uniform_ply_vertex_sample"}

    raise ValueError(f"unsupported ply format {fmt}: {path}")


def _parse_obj(path):
    vertices = []
    faces = []
    with Path(path).open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idxs = []
                for part in line.split()[1:]:
                    idx = int(part.split("/")[0])
                    idxs.append(idx - 1 if idx > 0 else len(vertices) + idx)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _sample_obj_surface(path, sample_points, seed):
    vertices, faces = _parse_obj(path)
    if len(vertices) == 0:
        raise ValueError(f"no vertices in obj: {path}")
    if len(faces) == 0:
        idx = _sample_indices(len(vertices), sample_points, seed)
        return vertices[idx], {"source_vertex_count": len(vertices), "source_face_count": 0, "method": "uniform_obj_vertex_sample"}

    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = areas > 0
    if not np.any(valid):
        idx = _sample_indices(len(vertices), sample_points, seed)
        return vertices[idx], {"source_vertex_count": len(vertices), "source_face_count": len(faces), "method": "uniform_obj_vertex_sample_zero_area_faces"}

    rng = np.random.default_rng(seed)
    probs = areas[valid].astype(np.float64)
    probs /= probs.sum()
    valid_faces = faces[valid]
    chosen = rng.choice(len(valid_faces), size=sample_points, replace=True, p=probs)
    chosen_tri = vertices[valid_faces[chosen]]
    u = rng.random(sample_points, dtype=np.float32)
    v = rng.random(sample_points, dtype=np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    pts = chosen_tri[:, 0] + u[:, None] * (chosen_tri[:, 1] - chosen_tri[:, 0]) + v[:, None] * (chosen_tri[:, 2] - chosen_tri[:, 0])
    return pts.astype(np.float32, copy=False), {
        "source_vertex_count": int(len(vertices)),
        "source_face_count": int(len(faces)),
        "method": "area_weighted_obj_surface_sample",
    }


def _write_binary_ply(path, pts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        pts.astype("<f4", copy=False).tofile(f)


def _cache_one_mesh(mesh_path, out_path, sample_points, seed, surface_sample_gt=False):
    suffix = Path(mesh_path).suffix.lower()
    if suffix == ".obj" and surface_sample_gt:
        pts, meta = _sample_obj_surface(mesh_path, sample_points, seed)
    elif suffix == ".ply":
        pts, meta = _sample_ply_vertices(mesh_path, sample_points, seed)
    elif suffix == ".obj":
        vertices, _ = _parse_obj(mesh_path)
        idx = _sample_indices(len(vertices), sample_points, seed)
        pts = vertices[idx]
        meta = {"source_vertex_count": int(len(vertices)), "method": "uniform_obj_vertex_sample"}
    else:
        raise ValueError(f"unsupported mesh format: {mesh_path}")

    _write_binary_ply(out_path, pts)
    st = Path(mesh_path).stat()
    meta.update({
        "source_mesh": str(Path(mesh_path).resolve()),
        "source_size_bytes": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
        "sampled_point_count": int(len(pts)),
        "seed": int(seed),
        "output": str(Path(out_path).resolve()),
    })
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2))
    return meta


def _condition_seed(base_seed, name):
    return int(base_seed + zlib.crc32(name.encode("utf-8"))) & 0xFFFFFFFF


def _run_conditions(args):
    index_path = Path(args.conditions_dir) / "conditions_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing conditions index: {index_path}")
    index = json.loads(index_path.read_text())
    if args.limit is not None:
        index = index[:args.limit]

    done = skipped = pruned = missing = 0
    for i, cond in enumerate(index, start=1):
        output = Path(cond["output_path"])
        out_path = output / args.sample_name
        if out_path.exists() and not args.force:
            skipped += 1
            if args.prune_mesh and (output / "mesh").exists():
                shutil.rmtree(output / "mesh")
                pruned += 1
            continue

        mesh_path = _latest_mesh(output / "mesh")
        if mesh_path is None:
            missing += 1
            print(f"[missing] {cond['condition']}: no mesh")
            continue

        seed = _condition_seed(args.seed, cond["condition"])
        meta = _cache_one_mesh(mesh_path, out_path, args.sample_points, seed)
        done += 1
        print(
            f"[{i}/{len(index)}] cached {cond['condition']} "
            f"points={meta['sampled_point_count']} method={meta['method']} -> {out_path}",
            flush=True,
        )
        if args.prune_mesh:
            shutil.rmtree(output / "mesh")
            pruned += 1

    print(f"done={done} skipped={skipped} missing={missing} pruned_mesh_dirs={pruned}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions-dir", help="Directory containing conditions_index.json")
    ap.add_argument("--mesh", help="Single mesh to sample, e.g. GT OBJ")
    ap.add_argument("--out", help="Output PLY for --mesh mode")
    ap.add_argument("--sample-points", type=int, default=100000)
    ap.add_argument("--sample-name", default="sampled_surface_100k.ply")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prune-mesh", action="store_true", help="Delete gsfusion_output/mesh after a cache exists.")
    ap.add_argument("--surface-sample-gt", action="store_true", help="Area-sample OBJ faces in --mesh mode.")
    args = ap.parse_args()

    if args.mesh:
        if not args.out:
            raise SystemExit("--out is required with --mesh")
        meta = _cache_one_mesh(args.mesh, args.out, args.sample_points, args.seed, args.surface_sample_gt)
        print(json.dumps(meta, indent=2))
        return

    if not args.conditions_dir:
        raise SystemExit("--conditions-dir is required unless --mesh is used")
    _run_conditions(args)


if __name__ == "__main__":
    main()
