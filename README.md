Python script to compare different photogrametric methods using coarse and fine registration.
Note : Unzip the meshes before referenceing in script

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/a9ed7404-73ea-4e37-99d1-5cb656d07a21" />

# Semi-automatic ICP with Open3D

Compare 3D reconstructions produced by different photogrammetric pipelines (e.g. **Agisoft Metashape** vs. **COLMAP + Poisson**) by registering one mesh onto the other with a coarse-to-fine alignment chain, then reporting overlap and residual error.

The script converts both meshes to point clouds, normalises them (center / scale / PCA orientation), computes a coarse transform either **automatically** (FPFH + RANSAC) or **manually** (interactive point picking), refines it with **point-to-plane ICP** using a robust Tukey kernel, and writes the aligned result to `aligned_source.ply`.

---

## Requirements

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 – 3.12 | Constrained by Open3D wheel availability |
| [Open3D](http://www.open3d.org/) | ≥ 0.17 | The robust-kernel block has fallbacks for 0.17, 0.17–0.18, and ≥ 0.18 APIs |
| [NumPy](https://numpy.org/) | ≥ 1.21 | |

### System

- A **desktop session with OpenGL support**. The script calls `o3d.visualization.draw_geometries(...)` and, in manual mode, `VisualizerWithEditing`. It will not run on a headless server or over a plain SSH session without an X server / virtual display.
- **RAM:** ~4 GB free is comfortable. Each mesh is sampled to 150,000 points, and the meshes themselves are loaded in full before sampling.

### Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install open3d numpy
```

Or, with a `requirements.txt`:

```text
open3d>=0.17
numpy>=1.21
```

```bash
pip install -r requirements.txt
```

---

## Input data

This repository ships the two meshes used in the example:

| File | Description |
|---|---|
| `Metashape-mesh.zip` | Source mesh — **unzip this first** to get `Metashape-mesh.obj` |
| `Colmap_poisson_mesh.obj` | Target mesh (COLMAP dense cloud → Poisson surface reconstruction) |
| `aligned_source.ply` | Example output — the source point cloud after registration |

> **Unzip `Metashape-mesh.zip` before running the script.** The script reads `.obj` files directly and will fail if the archive has not been extracted.

Both inputs must be **triangle meshes containing faces** (`f` lines in the OBJ). The script raises a `ValueError` if either mesh has zero triangles, since a face-less OBJ cannot be uniformly sampled.

---

## Usage

### 1. Set the mesh paths

The paths are currently **hard-coded absolute paths** near the top of `Semi-automatic_ICP.py` and point at the original author's machine. Edit them to your own locations before the first run:

```python
src_mesh = o3d.io.read_triangle_mesh(r"...\Metashape-mesh.obj")
tgt_mesh = o3d.io.read_triangle_mesh(r"...\Colmap_poisson_mesh.obj")
```

Since the script already does `os.chdir(os.path.dirname(__file__))`, the simplest fix is to make them relative to the repository:

```python
src_mesh = o3d.io.read_triangle_mesh("Metashape-mesh.obj")
tgt_mesh = o3d.io.read_triangle_mesh("Colmap_poisson_mesh.obj")
```

### 2. Run

```bash
python Semi-automatic_ICP.py
```

A viewer window opens at the end showing the result — **red = aligned source**, **green = target**. Close it to finish; the aligned cloud is written to `aligned_source.ply` in the repository directory.

### 3. (Optional) Manual initial alignment

For scenes where RANSAC fails — weak geometry, high symmetry, or very partial overlap — switch to interactive picking:

```python
use_manual = True    # near the top of section 3
```

You will be prompted twice, once per cloud:

1. **Ctrl + click** at least **3 corresponding points**, in the *same order* on both clouds.
2. Press **Q** to close the window and continue.

Picked indices are saved to `source_picked_points.txt` and `target_picked_points.txt`, and the initial transform is estimated point-to-point from those correspondences.

---

## How it works

| Stage | What happens |
|---|---|
| **0. Setup** | `chdir` to the script directory, print the Open3D version |
| **1. Load** | Read both OBJ meshes, compute vertex normals, sample 150,000 points uniformly from each |
| **2. Pre-align** | Translate both clouds to the origin, scale the source to match the target's bounding-box diagonal, then rotate it using PCA principal axes |
| **3. Coarse registration** | *Automatic:* voxel-downsample (0.03), estimate normals, compute FPFH features, run RANSAC with edge-length and distance checkers.<br>*Manual:* point-to-point transform from picked correspondences |
| **4. Fine registration** | Point-to-plane ICP with a Tukey robust kernel over the voxel/correspondence/iteration schedule in `icp_scales` |
| **5. Evaluate** | `evaluate_registration` at a 0.02 threshold → prints **fitness** (overlap fraction) and **inlier RMSE** |
| **6. Save** | Write the transformed source cloud to `aligned_source.ply` |

### Tunable parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `number_of_points` | Section 1 | `150000` | Sampling density; higher = slower, more detail |
| `use_manual` | Section 3 | `False` | Toggle manual picking vs. RANSAC |
| `voxel_size` | Section 3 | `0.03` | RANSAC downsampling; drives normal/feature radii |
| `icp_scales` | Section 4 | 0.06 / 0.03 / 0.015 | Per-level voxel size, max correspondence distance, iteration cap |
| evaluation threshold | Section 5 | `0.02` | Inlier distance for the reported fitness/RMSE |

**These defaults assume model units of roughly one metre in extent.** Photogrammetric output is often in arbitrary units — if fitness comes back near zero, the voxel sizes and correspondence distances are almost certainly the wrong order of magnitude for your data. Scale them to your scene before concluding the alignment failed.

---

## Output

```
Fitness (overlap): 0.812, RMSE: 0.004231
```

- **Fitness** — fraction of source points with a target point within the evaluation threshold. Higher is better; interpret it relative to the true overlap between the two reconstructions.
- **Inlier RMSE** — root-mean-square distance over those inlier correspondences, in model units. Lower is better.

Together these are the comparison metric: run the same pipeline against several photogrammetric reconstructions of the same subject and compare their fitness/RMSE against a common reference.

---

## Known limitations

- **Mesh paths are hard-coded** to absolute paths on the original author's machine and must be edited before the first run (see [Usage](#1-set-the-mesh-paths)).
- **The multi-scale ICP loop only executes once.** In section 4, the robust-kernel construction and the `registration_icp` call sit *outside* the `for lvl in icp_scales:` block. The loop body only downsamples and estimates normals, discarding each level; the single ICP run afterwards uses the loop variable left over from the final iteration. Effectively the alignment is a single-scale ICP at `voxel = 0.015, max_corr = 0.015, iters = 40`, not the intended coarse-to-fine cascade. Indenting those lines into the loop restores the intended behaviour.
- **`src.transform(T)` mutates in place.** `src_t` and `src` are the same object after section 5, so `src` is no longer in its pre-transform state if you extend the script.
- **PCA orientation is sign-ambiguous.** Eigenvectors are defined up to sign, so the pre-alignment rotation may flip an axis by 180°. RANSAC usually recovers from this; manual picking is the reliable fallback when it does not.
- Requires a GUI — no headless mode.

---

## Repository layout

```
.
├── Semi-automatic_ICP.py     # The registration pipeline
├── Metashape-mesh.zip        # Source mesh (unzip before use)
├── Colmap_poisson_mesh.obj   # Target mesh
├── aligned_source.ply        # Example output
└── README.md
```

---

## References

- Zhou, Q.-Y., Park, J., Koltun, V. — *Open3D: A Modern Library for 3D Data Processing* (2018)
- Rusu, R. B., Blodow, N., Beetz, M. — *Fast Point Feature Histograms (FPFH) for 3D Registration*, ICRA 2009
- Chen, Y., Medioni, G. — *Object modelling by registration of multiple range images*, Image and Vision Computing, 1992 (point-to-plane ICP)


