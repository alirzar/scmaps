"""Build the macaque cerebellar flatmap (src/scmaps/data/macaque/) from the
NMT v2.1-sym template and its SARM atlas, as AFNI distributes them.

The cerebellar cortex of NMT v2.1-sym -- the grey matter of its tissue
segmentation inside SARM's 21 level-6 cerebellar lobules per hemisphere -- is
unfolded lobule by lobule, as the lobules fan out around the cerebellum's
core in each sagittal plane:

* every grey-matter voxel of the left hemisphere gets its mediolateral
  position (x, mm) and its angle around the core of its sagittal slab -- the
  fastigial nucleus within CORE_MEDIAL mm of the midline, the centroid of the
  slab's white matter and deep nuclei beyond CORE_LATERAL mm, blended in
  between -- from lobule I (anterior) over the top (V, VI) and the back (VII,
  VIII) down to lobule X and the (para)flocculus;
* the angle is unwrapped by lobule (the anterior lobe above lobule I's
  start, lobules IX and X, the copula and the (para)flocculus below the
  posterior lobe) and made a length along the cerebellar surface by the
  median radius of the voxel's slab, smoothed over x;
* the map is a regular grid (GRID mm) over the points so placed, the right
  hemisphere the left's mirror image (the template is symmetric; the left
  half of the grid is computed, the right half copied); each vertex takes the
  majority lobule of its K nearest voxels, and a 3 x 3 mode filter removes
  speckle;
* the underlay is the depth below the cerebellum's outer envelope of the
  shallowest grey matter near each vertex (light on the folia, dark where a
  fissure opens), smoothed a little and given SUIT's underlay distribution
  (quantile matched to SUIT.shape.gii), so the shading reads as on the
  human flatmap.

The map keeps SARM's lobules, their order and their neighbours, and the
extent of the cerebellar surface. It does not unfold the folia: buried cortex
shares the place of the surface above it, so a lobule's flat area is that of
its outer envelope, not of its whole cortical sheet (the build prints both).
SARM's coarser levels (5, 4, 3) label each vertex by its lobule's parent
region, the level that holds most of the lobule's voxels.

Usage: python tools/build_macaque_flatmap.py NMT_ROOT [--out src/scmaps/data/macaque] [--preview fig.png]
       NMT_ROOT: the NMT v2.1-sym distribution (NMT_v2.1_sym/, tables_SARM/), e.g. from AFNI's @Install_NMT
"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CORTEX = ("Cb1V", "Cb2V", "Cb3V", "Cb4V", "Cb5V", "Cb6V", "Cb7V", "Cb8V", "Cb9V", "Cb10V",
          "Cb3I", "Cb4I", "Cb5I", "Cb6I", "Sim", "Crus1", "Crus2", "PM", "Cop", "PFl", "Fl")
TOP = {"Cb1V", "Cb2V", "Cb3V", "Cb4V", "Cb5V", "Cb3I", "Cb4I", "Cb5I"}     # the anterior lobe
BOTTOM = {"Cb9V", "Cb10V", "Cop", "PFl", "Fl"}                              # below the posterior lobe
NUCLEI = ("Fas", "IntA", "IntP", "Lat")
LEVELS = (3, 4, 5, 6)
GM, WM = 3, 4               # NMT v2.1 segmentation: grey matter, white matter
CORE_MEDIAL, CORE_LATERAL = 2.0, 5.0   # mm from the midline: the fastigial nucleus -> the slab's core
SLAB = 0.5                  # mm, the sagittal slabs of the core's centroid
RADIUS_SMOOTH = 1.0         # mm, the slab radius's smoothing over x
ENVELOPE = (0.5, 4.0)       # the outer envelope: the largest radius within +-0.5 mm in x and +-4 degrees
GRID = 0.25                 # mm, the flatmap's vertex spacing
REACH = 0.3                 # mm, a grid point within this of a voxel's flat position is on the map
K = 9                       # nearest voxels per vertex
UNDERLAY_SMOOTH = 1.0       # grid points, the depth's Gaussian smoothing (within the map)
EXTENT = 90.0               # flat units: the map's larger half-extent (SUIT's flatmap spans about +-90)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_key(path):
    """{index: (abbreviation, full name)} of a SARM key table."""
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 3 and f[0].strip().isdigit():
            out[int(f[0])] = (f[1].strip(), f[2].strip())
    return out


def core_centre(G, W, F):
    """(y, z) of the core for each x: the fastigial nucleus (F) near the midline, the slab centroid of the white
    matter and nuclei (W; a quadratic in x weighted by the slab's size, held where the white matter ends) laterally."""
    cx, cy, cz, cn = [], [], [], []
    for a in np.arange(np.floor(G[:, 0].min()), 0.0, SLAB):
        m = (W[:, 0] >= a) & (W[:, 0] < a + SLAB)
        if m.sum() >= 20:
            cx.append(a + SLAB / 2); cy.append(W[m, 1].mean()); cz.append(W[m, 2].mean()); cn.append(m.sum())
    cx, cy, cz, cn = map(np.asarray, (cx, cy, cz, cn))
    py, pz = np.polyfit(cx, cy, 2, w=np.sqrt(cn)), np.polyfit(cx, cz, 2, w=np.sqrt(cn))

    def centre(x):
        xc = np.clip(x, cx.min(), 0.0)
        w = np.clip((np.abs(x) - CORE_MEDIAL) / (CORE_LATERAL - CORE_MEDIAL), 0.0, 1.0)
        return (1 - w) * F[1] + w * np.polyval(py, xc), (1 - w) * F[2] + w * np.polyval(pz, xc)
    return centre


def unfold(G, lobule, centre):
    """Each voxel's flat position (x, v), its angle and its depth below the envelope (mm)."""
    x = G[:, 0]
    yc, zc = centre(x)
    t = np.degrees(np.arctan2(G[:, 2] - zc, G[:, 1] - yc)) % 360   # 0 anterior, 90 up, 180 back, 270 down
    r = np.hypot(G[:, 2] - zc, G[:, 1] - yc)
    name = np.asarray(CORTEX)[lobule]
    t[np.isin(name, sorted(TOP)) & (t >= 270)] -= 360
    t[np.isin(name, sorted(BOTTOM)) & (t < 90)] += 360
    slab = np.round(x / GRID).astype(int)
    keys, inv = np.unique(slab, return_inverse=True)
    med = np.array([np.median(r[inv == i]) for i in range(len(keys))])
    med = ndimage.gaussian_filter1d(med, RADIUS_SMOOTH / GRID, mode="nearest")
    v = -np.radians(t) * med[inv]                     # the anterior lobe at the top
    tree = cKDTree(np.c_[x / ENVELOPE[0], t / ENVELOPE[1]])
    env = np.array([r[n].max() for n in tree.query_ball_point(np.c_[x / ENVELOPE[0], t / ENVELOPE[1]], r=1.0)])
    return x, v, t, env - r


def mode3(lab, inside, n):
    """3 x 3 majority of the labels (0..n-1) over the inside points."""
    votes = np.stack([ndimage.uniform_filter(((lab == k) & inside).astype(float), 3, mode="constant") for k in range(n)])
    return np.where(inside, votes.argmax(0), -1)


def grid_mesh(inside):
    """Vertices (row, col) and counter-clockwise triangles of the inside grid points: two per inside cell, one per
    cell with three inside corners; the largest connected piece."""
    idx = -np.ones(inside.shape, int)
    rr, cc = np.nonzero(inside)
    idx[rr, cc] = np.arange(len(rr))
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]     # (r,c) (r,c+1) (r+1,c) (r+1,c+1)
    tri = []
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    tri += [np.c_[a[full], b[full], d[full]], np.c_[a[full], d[full], c[full]]]
    for p, q, s, miss in ((a, b, d, c), (a, d, c, b), (a, b, c, d), (b, d, c, a)):
        m = (p >= 0) & (q >= 0) & (s >= 0) & (miss < 0)
        tri.append(np.c_[p[m], q[m], s[m]])
    F = np.vstack(tri)
    n = len(rr)
    adj = coo_matrix((np.ones(3 * len(F)), (F.ravel(), np.roll(F, 1, axis=1).ravel())), shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    keep = comp == np.bincount(comp).argmax()
    new = -np.ones(n, int)
    new[keep] = np.arange(keep.sum())
    F = new[F]
    return np.c_[rr, cc][keep], F[(F >= 0).all(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nmt_root")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src" / "scmaps" / "data" / "macaque"))
    ap.add_argument("--preview")
    a = ap.parse_args()
    root, out = Path(a.nmt_root), Path(a.out)
    files = {"segmentation": root / "NMT_v2.1_sym" / "NMT_v2.1_sym_segmentation.nii.gz",
             "sarm": root / "NMT_v2.1_sym" / "SARM_in_NMT_v2.1_sym.nii.gz",
             **{f"key{L}": root / "tables_SARM" / f"SARM_key_{L}.txt" for L in LEVELS}}
    seg = np.asarray(nib.load(files["segmentation"]).dataobj).astype(int)
    img = nib.load(files["sarm"])
    sarm = np.asarray(img.dataobj).astype(int)                 # (x, y, z, 1, 6): SARM's six levels
    A = img.affine
    keys = {L: read_key(files[f"key{L}"]) for L in LEVELS}
    by_abbr = {L: {ab: i for i, (ab, _) in keys[L].items()} for L in LEVELS}
    l6 = sarm[..., 0, 5]
    left_ids = [by_abbr[6][f"SL_{c}"] for c in CORTEX]
    lob = np.isin(l6, left_ids)
    world = lambda m: np.argwhere(m) @ A[:3, :3].T + A[:3, 3]   # noqa: E731
    G = world(lob & (seg == GM))
    W = world((lob & (seg == WM)) | np.isin(l6, [by_abbr[6][f"SL_{n}"] for n in NUCLEI]))
    F = world(l6 == by_abbr[6]["SL_Fas"]).mean(0)
    pos = {i: k for k, i in enumerate(left_ids)}
    lobule = np.array([pos[i] for i in l6[lob & (seg == GM)]])
    x, v, t, depth = unfold(G, lobule, core_centre(G, W, F))
    # each lobule's parent at the coarser levels: the level's label of most of the lobule's voxels
    parent = {}
    for L in (3, 4, 5):
        lv = sarm[..., 0, L - 1]
        parent[L] = []
        for i in left_ids:
            ids, n = np.unique(lv[l6 == i], return_counts=True)
            parent[L].append(keys[L][int(ids[n.argmax()])][0][3:])
    # the grid, both hemispheres
    u, vv, lab, dep = np.r_[x, -x], np.r_[v, v], np.r_[lobule, lobule], np.r_[depth, depth]
    half = np.ceil((np.abs(u).max() + 1.0) / GRID) * GRID
    gu = np.arange(-half + GRID / 2, half, GRID)                 # no column on the midline
    gv = np.arange(np.floor(vv.min()) - 1.0, np.ceil(vv.max()) + 1.0 + GRID, GRID)
    U, V = np.meshgrid(gu, gv)
    P = np.c_[U.ravel(), V.ravel()]
    tree = cKDTree(np.c_[u, vv])
    d, nn = tree.query(P, k=K)
    mirror = lambda g: np.concatenate([g[:, :len(gu) // 2], g[:, :len(gu) // 2][:, ::-1]], axis=1)   # noqa: E731
    # the left half decides, the right half is its mirror image (nearest-voxel ties fall either way)
    inside = (d[:, 0] < REACH).reshape(U.shape)
    inside = mirror(ndimage.binary_fill_holes(ndimage.binary_closing(inside, iterations=2)))
    votes = lab[nn]
    maj = np.array([np.bincount(row, minlength=len(CORTEX)).argmax() for row in votes]).reshape(U.shape)
    maj = mirror(mode3(mirror(maj), inside, len(CORTEX)))
    shallow = np.where(inside, dep[nn].min(1).reshape(U.shape), 0.0)
    w = ndimage.gaussian_filter(inside.astype(float), UNDERLAY_SMOOTH)
    shallow = mirror(ndimage.gaussian_filter(shallow, UNDERLAY_SMOOTH) / np.maximum(w, 1e-9))   # within the map only
    suit = np.sort(np.asarray(nib.load(Path(__file__).resolve().parents[1] / "src" / "scmaps" / "data" /
                                       "SUIT.shape.gii").agg_data(), float))
    rank = np.argsort(np.argsort(-shallow[inside], kind="stable"), kind="stable") / max(inside.sum() - 1, 1)
    under = np.zeros(U.shape)
    under[inside] = np.interp(rank, np.linspace(0, 1, len(suit)), suit)            # SUIT's distribution
    under = mirror(under)                                                           # (ranks of equal depths)
    rc, faces = grid_mesh(inside)
    pu, pv = U[rc[:, 0], rc[:, 1]], V[rc[:, 0], rc[:, 1]]
    lob6 = maj[rc[:, 0], rc[:, 1]]
    if (lob6 < 0).any():
        raise SystemExit("a map vertex without a lobule")
    hemi = np.where(pu < 0, "L", "R")
    # flat units: centred, the larger half-extent EXTENT
    cu, cv = (pu.min() + pu.max()) / 2, (pv.min() + pv.max()) / 2
    s = EXTENT / max(pu.max() - cu, pv.max() - cv)
    pts = np.c_[(pu - cu) * s, (pv - cv) * s, np.zeros(len(pu))]
    out.mkdir(parents=True, exist_ok=True)
    surf = nib.gifti.GiftiImage()
    surf.add_gifti_data_array(nib.gifti.GiftiDataArray(pts.astype(np.float32), intent="NIFTI_INTENT_POINTSET",
                                                       datatype="NIFTI_TYPE_FLOAT32"))
    surf.add_gifti_data_array(nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE",
                                                       datatype="NIFTI_TYPE_INT32"))
    nib.save(surf, out / "FLAT.surf.gii")
    shape = nib.gifti.GiftiImage()
    shape.add_gifti_data_array(nib.gifti.GiftiDataArray(under[rc[:, 0], rc[:, 1]].astype(np.float32),
                                                        intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32"))
    nib.save(shape, out / "NMT.shape.gii")
    arrays = {"mm_per_unit": np.float64(1.0 / s),
              "inputs_sha256": json.dumps({k: sha256(p) for k, p in files.items()}, sort_keys=True)}
    full6 = {ab[3:]: name[3:].replace("_", " ") for ab, name in keys[6].values() if ab.startswith("SL_")}
    for L in LEVELS:
        regions = list(CORTEX) if L == 6 else list(dict.fromkeys(parent[L]))
        of = np.array([regions.index(c if L == 6 else parent[L][k]) for k, c in enumerate(CORTEX)])
        names = [f"S{h}_{r}" for h in "LR" for r in regions]
        labels = of[lob6] + 1 + np.where(hemi == "R", len(regions), 0)
        full = {ab[3:]: nm[3:].replace("_", " ") for ab, nm in keys[L].values() if ab.startswith("SL_")}
        arrays[f"labels_SARM{L}"] = labels.astype(np.uint8)
        arrays[f"names_SARM{L}"] = np.array(names)
        arrays[f"full_SARM{L}"] = np.array([f"{'left' if n[1] == 'L' else 'right'} {full.get(n[3:], full6.get(n[3:], n[3:]))}"
                                            for n in names])
    np.savez_compressed(out / "cerebellum_labels.npz", **arrays)
    # checks: every lobule on the map, in one piece; its flat area against its envelope and its grey matter
    print(f"flatmap: {len(pts)} vertices, {len(faces)} triangles, {1 / s:.4f} mm per flat unit")
    tri_area = 0.5 * np.abs(np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])[:, 2]) / s ** 2
    for k, c in enumerate(CORTEX):
        on = (lob6 == k) & (hemi == "L")
        f = faces[on[faces].all(1)]
        sub = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, axis=1).ravel())), shape=(len(pts),) * 2)
        ncomp = connected_components(sub, directed=False)[0] - (len(pts) - len(np.unique(f)))
        print(f"  {c:6s} flat {tri_area[on[faces].all(1)].sum():7.1f} mm2  grey matter {np.sum(lobule == k) * 0.25 ** 3:7.1f} mm3"
              f"  pieces {ncomp}  SARM5 {parent[5][k]}  SARM4 {parent[4][k]}  SARM3 {parent[3][k]}")
    print("->", out)
    if a.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=110)
        cm = matplotlib.colormaps["tab20"]
        col = np.array([cm(k % 20)[:3] for k in range(len(CORTEX))])
        col[20] = (0.3, 0.3, 0.3)
        ax[0].tripcolor(pts[:, 0], pts[:, 1], faces, facecolors=lob6[faces[:, 0]], cmap=matplotlib.colors.ListedColormap(col),
                        vmin=-0.5, vmax=len(CORTEX) - 0.5)
        for k, c in enumerate(CORTEX):
            m = (lob6 == k) & (hemi == "L")
            ax[0].text(np.median(pts[m, 0]), np.median(pts[m, 1]), c, ha="center", va="center", fontsize=6.5,
                       bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.4))
        ax[1].tripcolor(pts[:, 0], pts[:, 1], faces, under[rc[:, 0], rc[:, 1]], cmap="gray", vmin=-1, vmax=0.5, shading="gouraud")
        for x_ in ax:
            x_.set_aspect("equal"); x_.set_axis_off()
        fig.savefig(a.preview, bbox_inches="tight")
        print("preview ->", a.preview)


if __name__ == "__main__":
    main()
