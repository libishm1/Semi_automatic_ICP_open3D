# Semi-automatic ICP with Open3D

Coarse-to-fine registration of two 3D surfaces, for comparing reconstructions of the same subject produced by different pipelines — photogrammetry (Metashape vs. COLMAP), intraoral/desktop scanners, or repeat scans of the same object.

<img width="400" height="400" alt="Registration result" src="https://github.com/user-attachments/assets/a9ed7404-73ea-4e37-99d1-5cb656d07a21" />

`Semi-automatic_ICP.py` samples both surfaces to point clouds, centres them, estimates a coarse transform (FPFH + RANSAC, or manual point picking), refines it with a four-level robust point-to-plane ICP cascade, and reports the residual surface deviation.

Every length parameter is a **fraction of the bounding-box diagonal**, so the same defaults work in metres, millimetres or arbitrary units.

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 – 3.12 | **Open3D has no wheel for Python 3.13** — a 3.13 install fails at `pip install open3d` |
| [Open3D](http://www.open3d.org/) | ≥ 0.17 | Verified on 0.19.0 |
| [NumPy](https://numpy.org/) | ≥ 1.21 | Verified on 2.4.6 |

```bash
# If your default python is 3.13, pick 3.11 or 3.12 explicitly:
py -3.11 -m venv .venv          # Windows
python3.11 -m venv .venv        # macOS / Linux

.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

A desktop session with OpenGL is needed for the viewer and for `--manual`. Everything else runs headless with `--no-vis`.

### Hardware

**An 8 GB laptop is comfortable.** Measured peak working set on the 20 MB / 1.3M-vertex intraoral scan pair: **663 MB**, of which 101 MB is just importing NumPy and Open3D. With threads capped to 4 (a typical 11th-gen i5 laptop) the scans took 2.3 s and the photogrammetry meshes 8.4 s.

| Stage | Resident | Peak |
|---|---|---|
| Imports only | 101 MB | 101 MB |
| `read_triangle_mesh` on a 20 MB STL | 169 MB | **611 MB** |
| `compute_vertex_normals` | 179 MB | 611 MB |
| Sample 150k points | 187 MB | 611 MB |
| Second mesh loaded + sampled | 266 MB | 663 MB |
| Both meshes released | 119 MB | 663 MB |
| ICP + KD-tree distance queries | 120 MB | 663 MB |

- **Peak is set by the mesh reader, not the registration.** Parsing an STL costs a transient of roughly **25× the file size**, which then settles back. Normals, sampling, FPFH, ICP and KD-trees are comparatively free.
- **Lowering `--points` does not reduce peak memory.** Peak was identical at 75k, 200k and 400k points, because the spike happens before sampling. Lower it to make the viewer smoother, not to save RAM.

Sizing rule: budget about **25× your largest single mesh file**, plus ~150 MB. On 8 GB with Windows using ~3 GB, mesh files up to roughly 150 MB are safe; beyond that, decimate first.

---

## Usage

```bash
python Semi-automatic_ICP.py SOURCE TARGET [options]
```

```bash
# Two metric scans, headless, results into a folder
python Semi-automatic_ICP.py scan_a.stl scan_b.stl --no-vis --out-dir results

# Inputs genuinely in different units (e.g. arbitrary photogrammetric scale)
python Semi-automatic_ICP.py mesh_a.obj mesh_b.obj --allow-scale

# RANSAC failed? Pick correspondences by hand (Shift+click, then Q)
python Semi-automatic_ICP.py scan_a.stl scan_b.stl --manual
```

Reads STL, OBJ, PLY, OFF, GLTF — anything Open3D's mesh reader accepts. Faceless files are retried as point clouds.

| Option | Default | Purpose |
|---|---|---|
| `--points N` | `200000` | Points sampled from each mesh |
| `--voxel-frac F` | `0.015` | Base voxel size as a fraction of the bbox diagonal |
| `--allow-scale` | off | Permit resizing the source. **Off by default** — see below |
| `--no-pca` | off | Skip the PCA orientation guess |
| `--manual` | off | Pick ≥3 correspondences interactively |
| `--no-ransac` | off | Skip the coarse stage; start ICP from the pre-alignment |
| `--no-vis` | off | Headless — never open a window |
| `--out-dir D` | `.` | Where results are written |
| `--seed N` | `42` | Seeds Open3D's global RNG (sampling and RANSAC) |

> `--seed` changes the random draw but does **not** give bit-identical runs — Open3D's parallel sampling and RANSAC are not fully deterministic. Repeat runs at a fixed seed varied by about 0.6 % in mean deviation.

### Outputs

| File | Contents |
|---|---|
| `aligned_source.ply` | Source point cloud in target coordinates |
| `deviation_heatmap.ply` | Same cloud, coloured blue (0) → red (0.5 % of diagonal) |
| `transform.txt` | 4×4 matrix mapping **original** source coords → **original** target coords |
| `report.json` | Every parameter, per-level ICP fitness, and all deviation statistics |

---

## Scale is locked by default

Rescaling the source so the two bounding-box diagonals match is reasonable for photogrammetry in arbitrary units. For metrically-calibrated scans it destroys the very thing you are measuring — and a bounding-box diagonal responds to *coverage and pose*, not just scale, so two scans of the same object trimmed differently have different diagonals despite being metrically identical.

### Do the two example datasets need different settings?

**Exactly one flag differs.** Every length is scale-relative, so `--voxel-frac 0.015` becomes 0.049 units on the photogrammetry meshes and 2.68 mm on the scans, automatically.

| | Photogrammetry (`.obj`) | Metric scans (`.stl`) |
|---|---|---|
| Units | arbitrary | millimetres |
| Diagonals | 2.14 vs 3.26 — **53 % apart** | 182.4 vs 178.7 — 2.1 % apart |
| ...because | genuinely different scale | different trimming/coverage |
| Flag | `--allow-scale` | *(omit it)* |

The distinction is *why* the diagonals differ, not by how much. On the scans, wrongly enabling scale applies a 2.1 % shrink and drops overlap from 90 % to 38 %:

| Scans | mean | rms | p95 |
|---|---|---|---|
| Rigid (default) | **0.198 mm** | **0.222 mm** | **0.364 mm** |
| `--allow-scale` (wrong here) | 0.613 mm | 1.045 mm | 3.341 mm |

Rule of thumb: **if both inputs came off calibrated hardware, lock the scale.** Only unlock it for reconstructions with no absolute reference, such as photogrammetry without a scale bar.

---

## Worked example

Two fused intraoral scans (`iTAD-scans.zip`), ~435k and ~406k triangles, roughly 180 mm across, starting 120 mm and 17° apart.

```
python Semi-automatic_ICP.py Fused_20260721113854.stl Fused_20260721112631.stl --no-vis --out-dir results
```

```
200,000 / 200,000 points; diagonals 182.5027 / 178.7141 (ratio 1.0212)
NOTE: diagonals differ by +2.1%; scale locked (rigid).
working diagonal 178.7141 -> base voxel 2.6807
RANSAC: fitness 1.0000, rmse 1.10427
  ICP 1  voxel   10.7247  pts     227  fitness 1.0000  rmse 4.32454
  ICP 2  voxel    5.3624  pts     897  fitness 1.0000  rmse 1.84883
  ICP 3  voxel    2.6812  pts   3,427  fitness 1.0000  rmse 1.06257
  ICP 4  voxel    1.3406  pts  13,104  fitness 0.9995  rmse 0.52596

SURFACE DEVIATION (aligned source -> nearest target point)
    mean :    0.19828   ( 0.111 % of diagonal)
  median :    0.18651   ( 0.104 % of diagonal)
     rms :    0.22165   ( 0.124 % of diagonal)
     p95 :    0.36416   ( 0.204 % of diagonal)
     max :    2.67361   ( 1.496 % of diagonal)
  within 0.001xdiag 46.7%, 0.002xdiag 94.4%, 0.005xdiag 99.9%
```

Total runtime **2.6 s**. Independent verification, recomputed from the original coordinates:

```
det(R) = 1.0000000010     singular values of R = [1. 1. 1.]   (exactly rigid, no scaling)
rotation 17.229 deg       translation 120.248 mm

before alignment: mean 100.1417 mm   after: mean 0.1981 mm    (505x improvement)
reverse (target -> source): mean 0.1991 mm, p95 0.3679 mm
```

Forward and reverse deviations agreeing to within 0.001 mm indicates a symmetric, full-overlap fit rather than one surface fitting into part of the other.

The photogrammetry pair (`Metashape-mesh.zip` vs `Colmap_poisson_mesh.obj`) runs in 1.8 s with `--allow-scale`, improving mean deviation 688×.

### Reading the numbers

- **Fitness** — fraction of points with a correspondence inside the threshold. Near 1.0 means the overlap was found; it says nothing about how *well* it fits.
- **Deviation stats** — the actual agreement, in model units. This is the metrology result.
- **p95 and max** — where the surfaces genuinely disagree. On repeat scans this is usually deformable tissue, trimming differences, or scanner noise at grazing angles.

Watch for **high fitness with high RMSE**: correspondences were found, but the surfaces do not really match.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for open3d` | Python 3.13 | Use Python 3.11 or 3.12 |
| RANSAC fitness ≈ 0, near-identity transform | Voxel wrong for the units | Handled by `--voxel-frac`; raise it if the scans are noisy |
| `voxel ... left too few points` | `--voxel-frac` too coarse | Lower it (try `0.005`) |
| Converged to a wrong pose | Symmetric or low-feature geometry | `--manual`, or try a different `--seed` |
| Fit worse than expected | Scale wrongly enabled | Drop `--allow-scale` |
| Crash on a headless machine | Viewer needs OpenGL | Add `--no-vis` |
| `MemoryError` / heavy paging while loading | Mesh too large for available RAM | Decimate first — lowering `--points` will not help, the spike is in the reader |
| Viewer sluggish on integrated graphics | Too many points rendered | `--points 50000`; affects display only, not the transform |

---

## History

This started as a fixed-path script with metre-scale constants. Run on millimetre-scale intraoral scans it produced **fitness 0.0000 after 7 min 16 s**; the same data now completes in 2.6 s. What was wrong, all measured:

1. **Metre-scale constants.** `voxel_size = 0.03` against a 180 mm model. `voxel_down_sample(0.03)` reduced 150,000 points to 149,467 — a 0.4 % reduction instead of the intended ~50×, so RANSAC ran on 149k points.
2. **Degenerate FPFH features.** A feature radius of 0.15 mm against ~0.48 mm point spacing left **82,216 of 149,467 descriptors (55 %) all-zero**, so RANSAC returned the identity matrix.
3. **The multi-scale ICP loop never looped.** The kernel construction and the `registration_icp` call sat *outside* `for lvl in icp_scales:` — the loop only downsampled and estimated normals, and the single ICP afterwards ran on the leaked loop variable.
4. **Unconditional rescaling** broke metric comparison.
5. **PCA sign ambiguity.** Eigenvectors are defined up to sign, so the pre-alignment could flip an axis by 180° between runs. Axes are now oriented by third-moment skew with `det = +1` enforced.
6. **`src.transform(T)` mutated in place**, leaving the source and its transformed copy as the same object.
7. **Hard-coded absolute paths**, now command-line arguments.
8. **Manual mode raised `TypeError`.** `compute_transformation()` requires a third `corres` argument, so the "semi-automatic" half of the script had never worked on this Open3D version. The on-screen hint also said Ctrl+Click; Open3D uses **Shift+Click**.

---

## Repository layout

```
.
├── Semi-automatic_ICP.py     # The whole tool (~200 lines)
├── requirements.txt
├── iTAD-scans.zip            # Two intraoral scans, millimetres (rigid example)
├── Metashape-mesh.zip        # Photogrammetry source (arbitrary units)
├── Colmap_poisson_mesh.obj   # Photogrammetry target
└── aligned_source.ply        # Example output
```

> Unzip the archives before pointing the script at them.

---

## References

- Zhou, Q.-Y., Park, J., Koltun, V. — *Open3D: A Modern Library for 3D Data Processing*, 2018
- Rusu, R. B., Blodow, N., Beetz, M. — *Fast Point Feature Histograms (FPFH) for 3D Registration*, ICRA 2009
- Chen, Y., Medioni, G. — *Object modelling by registration of multiple range images*, Image and Vision Computing, 1992
