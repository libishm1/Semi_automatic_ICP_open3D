#!/usr/bin/env python3
"""Semi-automatic coarse-to-fine ICP registration of two 3D surfaces.

Sample -> centre -> [PCA] -> coarse (FPFH+RANSAC or manual picking) ->
multi-scale robust point-to-plane ICP -> deviation report.

Every length is a fraction of the bounding-box diagonal, so the same defaults
work in metres, millimetres or arbitrary photogrammetric units. Scale is locked
(rigid) unless --allow-scale is given; rescaling calibrated scans destroys the
accuracy you are trying to measure.

  python Semi-automatic_ICP.py a.stl b.stl --no-vis --out-dir results
  python Semi-automatic_ICP.py a.obj b.obj --allow-scale   # differing units
  python Semi-automatic_ICP.py a.stl b.stl --manual        # pick by hand
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d

REG = o3d.pipelines.registration
KDT = o3d.geometry.KDTreeSearchParamHybrid


def load(path, n):
    """Read a mesh (or point cloud) and return n sampled points."""
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles):
        mesh.compute_vertex_normals()
        return mesh.sample_points_uniformly(n)
    pcd = o3d.io.read_point_cloud(path)          # faceless OBJ / raw cloud
    if not len(pcd.points):
        raise SystemExit(f"{path}: no triangles and no points.")
    return pcd.uniform_down_sample(max(1, len(pcd.points) // n))


def diag(pcd):
    return float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))


def pca(pcd):
    """Principal axes, sign-fixed by third-moment skew so runs are repeatable."""
    v = np.asarray(pcd.points)
    v = v - v.mean(0)
    ax = np.linalg.eigh(np.cov(v.T))[1][:, ::-1]          # descending eigenvalue
    ax = ax * np.where(((v @ ax) ** 3).sum(0) < 0, -1.0, 1.0)
    if np.linalg.det(ax) < 0:
        ax[:, 2] *= -1
    return ax


def prep(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(KDT(radius=voxel * 2, max_nn=30))
    return down


def tukey(k):
    """Tukey robust kernel, across Open3D API generations."""
    try:
        return REG.TukeyLoss(k)                            # >= 0.18
    except AttributeError:
        return REG.RobustKernel(REG.RobustKernelType.Tukey, k)


def pick(pcd, title):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    print(f"  {title}: Shift+Click at least 3 points in order, then press Q.")
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    if len(idx) < 3:
        raise SystemExit(f"only {len(idx)} point(s) picked; at least 3 needed.")
    return np.asarray(pcd.points)[idx]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="mesh/cloud to move")
    ap.add_argument("target", help="reference mesh/cloud")
    ap.add_argument("--points", type=int, default=200000, help="points sampled per mesh")
    ap.add_argument("--voxel-frac", type=float, default=0.015, help="base voxel / bbox diagonal")
    ap.add_argument("--allow-scale", action="store_true", help="permit resizing (breaks metric accuracy)")
    ap.add_argument("--no-pca", action="store_true", help="skip PCA orientation guess")
    ap.add_argument("--manual", action="store_true", help="pick correspondences by hand")
    ap.add_argument("--no-ransac", action="store_true", help="skip coarse stage")
    ap.add_argument("--no-vis", action="store_true", help="headless")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="RANSAC seed")
    o = ap.parse_args()
    os.makedirs(o.out_dir, exist_ok=True)
    # Seeds Open3D's global RNG, which drives both point sampling and RANSAC.
    # (registration_ransac_* takes no `seed` argument of its own.)
    o3d.utility.random.seed(o.seed)
    print(f"Open3D {o3d.__version__}")

    src, tgt = load(o.source, o.points), load(o.target, o.points)
    ds, dt = diag(src), diag(tgt)
    print(f"{len(src.points):,} / {len(tgt.points):,} points; "
          f"diagonals {ds:.4f} / {dt:.4f} (ratio {ds / dt:.4f})")

    # Pre-alignment, tracked in P so the final transform maps ORIGINAL source
    # coordinates to ORIGINAL target coordinates.
    cs = src.get_axis_aligned_bounding_box().get_center()
    ct = tgt.get_axis_aligned_bounding_box().get_center()
    src.translate(-cs)
    tgt.translate(-ct)
    P = np.eye(4)
    P[:3, 3] = -cs

    scale = 1.0
    if o.allow_scale:
        scale = dt / ds
        src.scale(scale, center=(0, 0, 0))
        P[:3] *= scale
        print(f"applied scale {scale:.6f} ({(scale - 1) * 100:+.2f}%)")
    elif abs(ds / dt - 1) > 0.02:
        print(f"NOTE: diagonals differ by {(ds / dt - 1) * 100:+.1f}%; scale locked "
              "(rigid). Use --allow-scale only if the inputs are in different units.")

    if not o.no_pca:
        M = np.eye(4)
        M[:3, :3] = pca(tgt) @ pca(src).T
        src.rotate(M[:3, :3], center=(0, 0, 0))
        P = M @ P

    d = diag(tgt)
    base = d * o.voxel_frac
    print(f"working diagonal {d:.4f} -> base voxel {base:.4f}")

    # --- coarse ---------------------------------------------------------
    if o.manual:
        if o.no_vis:
            raise SystemExit("--manual needs a viewer; drop --no-vis.")
        sp, tp = pick(src, "SOURCE"), pick(tgt, "TARGET")
        n = min(len(sp), len(tp))
        corr = o3d.utility.Vector2iVector(np.tile(np.arange(n), (2, 1)).T)
        T = REG.TransformationEstimationPointToPoint(False).compute_transformation(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sp[:n])),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp[:n])), corr)
    elif o.no_ransac:
        T = np.eye(4)
    else:
        sd, td = prep(src, base), prep(tgt, base)
        if min(len(sd.points), len(td.points)) < 100:
            raise SystemExit(f"voxel {base:.4f} left too few points; lower --voxel-frac.")
        feat = lambda p: REG.compute_fpfh_feature(p, KDT(radius=base * 5, max_nn=100))
        lim = base * 1.5
        res = REG.registration_ransac_based_on_feature_matching(
            sd, td, feat(sd), feat(td), mutual_filter=True,
            max_correspondence_distance=lim,
            estimation_method=REG.TransformationEstimationPointToPoint(False), ransac_n=4,
            checkers=[REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      REG.CorrespondenceCheckerBasedOnDistance(lim)],
            criteria=REG.RANSACConvergenceCriteria(1000000, 0.999))
        T = res.transformation
        print(f"RANSAC: fitness {res.fitness:.4f}, rmse {res.inlier_rmse:.5f}")

    # --- fine: genuine coarse-to-fine cascade ---------------------------
    levels = []
    for i, mul in enumerate([4.0, 2.0, 1.0, 0.5], 1):
        voxel, lim = base * mul, base * mul * 1.5
        sl, tl = prep(src, voxel), prep(tgt, voxel)
        if min(len(sl.points), len(tl.points)) < 50:
            continue
        res = REG.registration_icp(
            sl, tl, lim, init=T,
            estimation_method=REG.TransformationEstimationPointToPlane(tukey(lim * 0.5)),
            criteria=REG.ICPConvergenceCriteria(max_iteration=70 - 10 * i))
        T = res.transformation
        levels.append({"voxel": voxel, "points": len(sl.points),
                       "fitness": res.fitness, "rmse": res.inlier_rmse})
        print(f"  ICP {i}  voxel {voxel:9.4f}  pts {len(sl.points):7,}  "
              f"fitness {res.fitness:.4f}  rmse {res.inlier_rmse:.5f}")
    if not levels:
        raise SystemExit("every ICP level was skipped; check --voxel-frac.")

    # --- evaluate -------------------------------------------------------
    moved = o3d.geometry.PointCloud(src)          # copy: leave src untransformed
    moved.transform(T)
    dist = np.asarray(moved.compute_point_cloud_distance(tgt))
    stats = {"mean": dist.mean(), "median": np.median(dist), "rms": np.sqrt((dist ** 2).mean()),
             "p95": np.percentile(dist, 95), "p99": np.percentile(dist, 99), "max": dist.max()}
    print("\nSURFACE DEVIATION (aligned source -> nearest target point)")
    for k, v in stats.items():
        print(f"  {k:>6} : {v:10.5f}   ({v / d * 100:6.3f} % of diagonal)")
    within = {f"{f:g}xdiag": float((dist <= d * f).mean()) for f in (0.001, 0.002, 0.005)}
    print("  within " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in within.items()))

    B = np.eye(4)
    B[:3, 3] = ct
    full = B @ T @ P
    print("\nTransform (original source -> original target coords):")
    print(np.array2string(full, precision=6, suppress_small=True))

    # --- save -----------------------------------------------------------
    out = lambda f: os.path.join(o.out_dir, f)
    o3d.io.write_point_cloud(out("aligned_source.ply"), moved)
    heat = o3d.geometry.PointCloud(moved)
    t = np.clip(dist / (d * 0.005), 0, 1)         # blue = 0, red >= 0.5% of diagonal
    heat.colors = o3d.utility.Vector3dVector(np.c_[t, 1 - abs(2 * t - 1), 1 - t])
    o3d.io.write_point_cloud(out("deviation_heatmap.ply"), heat)
    np.savetxt(out("transform.txt"), full, fmt="%.9f")
    json.dump({"source": os.path.abspath(o.source), "target": os.path.abspath(o.target),
               "open3d": o3d.__version__, "points": o.points, "voxel_frac": o.voxel_frac,
               "base_voxel": base, "diagonal": d, "scale": scale,
               "scale_locked": not o.allow_scale, "icp_levels": levels,
               "deviation": {k: float(v) for k, v in stats.items()}, "within": within,
               "transform": full.tolist()}, open(out("report.json"), "w"), indent=2)
    print(f"\nwrote aligned_source.ply, deviation_heatmap.ply, transform.txt, "
          f"report.json -> {os.path.abspath(o.out_dir)}")

    if not o.no_vis:
        print("Red = aligned source, green = target. Close the window to continue.")
        o3d.visualization.draw_geometries(
            [o3d.geometry.PointCloud(moved).paint_uniform_color([0.85, 0.15, 0.15]),
             o3d.geometry.PointCloud(tgt).paint_uniform_color([0.15, 0.75, 0.15])],
            window_name="Registration result")
        o3d.visualization.draw_geometries([heat], window_name="Deviation heat map")


if __name__ == "__main__":
    main()
