
import open3d as o3d
import numpy as np
import os
import sys

# -----------------------------------------------------------------
# 0️⃣ Setup: working directory & version info
# -----------------------------------------------------------------
os.chdir(os.path.dirname(__file__))
print("📂 Working directory:", os.getcwd())

print("🔹 Using Open3D version:", o3d.__version__)

# -----------------------------------------------------------------
# 1️⃣ Load OBJ meshes and convert to point clouds
# -----------------------------------------------------------------
print("🔹 Loading OBJ files...")

src_mesh = o3d.io.read_triangle_mesh(
    r"C:\Users\lmurugesan\OneDrive - Alfaisal University\Documents\Open-3d\Open3D-main\Open3D-main\examples\python\ICP\Metashape-mesh.obj"
)
tgt_mesh = o3d.io.read_triangle_mesh(
    r"C:\Users\lmurugesan\OneDrive - Alfaisal University\Documents\Open-3d\Open3D-main\Open3D-main\examples\python\ICP\Colmap_poisson_mesh.obj"
)

src_mesh.compute_vertex_normals()
tgt_mesh.compute_vertex_normals()

if len(src_mesh.triangles) == 0 or len(tgt_mesh.triangles) == 0:
    raise ValueError("❌ One or both OBJ meshes have no triangles. "
                     "Ensure your .obj files contain 'f' (faces).")

src = src_mesh.sample_points_uniformly(number_of_points=150000)
tgt = tgt_mesh.sample_points_uniformly(number_of_points=150000)
print("✅ Converted OBJ meshes to point clouds.")

# -----------------------------------------------------------------
# 2️⃣ Pre-alignment (center, scale, orientation)
# -----------------------------------------------------------------
bbox_src, bbox_tgt = src.get_axis_aligned_bounding_box(), tgt.get_axis_aligned_bounding_box()
center_src, center_tgt = bbox_src.get_center(), bbox_tgt.get_center()

src.translate(-center_src)
tgt.translate(-center_tgt)

scale_factor = np.linalg.norm(bbox_tgt.get_extent()) / np.linalg.norm(bbox_src.get_extent())
src.scale(scale_factor, center=(0, 0, 0))

def pca_axes(pcd):
    pts = np.asarray(pcd.points)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]

R = pca_axes(tgt) @ pca_axes(src).T
src.rotate(R, center=(0, 0, 0))
print("✅ Pre-alignment (center, scale, orientation) done.")

# -----------------------------------------------------------------
# 3️⃣ Manual picking (optional) or automatic RANSAC
# -----------------------------------------------------------------
use_manual = False  # set True to enable point picking

def pick_points(pcd, filename):
    save_path = os.path.join(os.path.dirname(__file__), filename)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick points for {filename}")
    vis.add_geometry(pcd)
    print("👉 Use Ctrl+Click to pick ≥3 points. Press Q to finish.")
    vis.run()
    vis.destroy_window()
    if os.path.exists(filename) and not os.path.samefile(filename, save_path):
        os.replace(filename, save_path)
    print(f"✅ Saved picked points to {save_path}")

def load_picks(pcd, fn):
    with open(fn, "r") as f:
        ids = [int(x.strip()) for x in f.readlines()]
    if not ids:
        raise ValueError(f"⚠️ No points picked in {fn}.")
    return np.asarray(pcd.points)[ids]

if use_manual:
    print("\nPick ≥3 corresponding points from SOURCE (Ctrl+click → Q).")
    pick_points(src, "source_picked_points.txt")
    print("Pick same-number corresponding points from TARGET (Ctrl+click → Q).")
    pick_points(tgt, "target_picked_points.txt")

    src_pts = load_picks(src, "source_picked_points.txt")
    tgt_pts = load_picks(tgt, "target_picked_points.txt")

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T_init = est.compute_transformation(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts))
    )
    print("✅ Manual initial transform computed.")
else:
    print("\nRunning automatic global registration (FPFH + RANSAC)...")
    voxel_size = 0.03
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    distance_threshold_ransac = voxel_size * 1.5

    def preprocess(pcd, voxel):
        p = pcd.voxel_down_sample(voxel)
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        return p

    src_d = preprocess(src, voxel_size)
    tgt_d = preprocess(tgt, voxel_size)

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )

    T_init = result.transformation
    print("✅ Automatic coarse transform computed via RANSAC.")

print("Initial transform:\n", T_init)

# -----------------------------------------------------------------
# 4️⃣ Multi-scale ICP (robust, version-safe)
# -----------------------------------------------------------------
print("\nRunning fine multi-scale ICP (point-to-plane, robust)...")

icp_scales = [
    {"voxel": 0.06, "max_corr": 0.045, "iters": 60},
    {"voxel": 0.03, "max_corr": 0.03, "iters": 50},
    {"voxel": 0.015, "max_corr": 0.015, "iters": 40},
]

T = T_init.copy()

for lvl in icp_scales:
    src_l = src.voxel_down_sample(lvl["voxel"])
    tgt_l = tgt.voxel_down_sample(lvl["voxel"])

    src_l.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=lvl["voxel"] * 2, max_nn=30)
    )
    tgt_l.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=lvl["voxel"] * 2, max_nn=30)
    )
# --- Robust kernel (fully version-safe) ---
try:
    # ✅ Open3D ≥ 0.18.0 (new loss-class API)
    kernel = o3d.pipelines.registration.TukeyLoss(lvl["max_corr"])
except AttributeError:
    try:
        # ✅ Open3D 0.17–0.18 transitional API
        kernel = o3d.pipelines.registration.RobustKernel("tukey", lvl["max_corr"])
    except Exception:
        # ✅ Open3D ≤ 0.17 legacy API
        kernel = o3d.pipelines.registration.RobustKernel(
            o3d.pipelines.registration.RobustKernelType.Tukey,
            lvl["max_corr"]
        )

estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=lvl["iters"])
reg = o3d.pipelines.registration.registration_icp(
        src_l, tgt_l, lvl["max_corr"], init=T,
        estimation_method=estimation, criteria=criteria
    )

T = reg.transformation

print("✅ Refined ICP transform:\n", T)

# -----------------------------------------------------------------
# 5️⃣ Evaluate and visualize
# -----------------------------------------------------------------
eval_result = o3d.pipelines.registration.evaluate_registration(src, tgt, 0.02, T)
print(f"Fitness (overlap): {eval_result.fitness:.3f}, RMSE: {eval_result.inlier_rmse:.6f}")

src.paint_uniform_color([1, 0, 0])  # red
tgt.paint_uniform_color([0, 1, 0])  # green
src_t = src.transform(T.copy())

print("\n✅ Alignment complete. Red = aligned source, Green = target.")
o3d.visualization.draw_geometries([src_t, tgt])

# -----------------------------------------------------------------
# 6️⃣ Save output
# -----------------------------------------------------------------
o3d.io.write_point_cloud("aligned_source.ply", src_t)
print("💾 Saved aligned source as 'aligned_source.ply'")
