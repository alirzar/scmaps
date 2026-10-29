"""Build the macaque subcortical surfaces (src/scmaps/data/macaque/subcortex.npz)
from the NMT v2.1-sym template's SARM atlas, as AFNI distributes it.

Every SARM region of levels 3 to 6 but the cerebellum's (its cortex is on the
cerebellar flatmap; its nuclei and white matter would cover the brainstem) and
the pituitary, as a surface for the views scmaps draws the human subcortex in.
The human surfaces are MATLAB's boundary(..., 0.5) of Tian S2's 2 mm voxel
centres: simple shapes, about 200 voxels a region, two touching regions one
voxel apart. Here:

* the labels on a GRID mm grid (the 0.25 mm template coarsened 2x, each
  2x2x2 block its most frequent label, a tie going to a region over the
  background), which gives SARM's regions about as many voxels as the human
  map's (level 3: median 1,000; level 5: 150); on a 1 mm grid level 5's
  nuclei break into shards and level 3's lateral pallium, a thin sheet over
  the striatum, into patches;
* surface nets (VTK's vtkSurfaceNets3D) over every region of the level at
  once, each region's faces taken out of it and turned outward; unlike
  boundary(), whose alpha shape fills concavities (the striatum, the fornix
  and the reticular thalamus become blobs), it keeps the regions' shapes;
* windowed-sinc smoothing (SINC_ITERATIONS, PASSBAND) and quadric decimation
  to KEEP of the triangles (at least MIN_FACES a region);
* each surface pulled in half a voxel along its normal, never more than CAP
  of the region's local half-thickness (that estimate averaged over the
  surface), so two touching regions sit a voxel apart, as on the human map,
  and thin ones thin out instead of turning inside out;
* a region too small for the grid built alone on the template's 0.25 mm grid;
* no left vertex beyond x = MIDLINE_X (smoothing can carry a midline region's
  rim across the midline, where its right half would meet it).

Only the left hemisphere is built: the template and SARM are exactly symmetric
(checked here: the volume mirrored at x = 0 with SL_ and SR_ swapped is the
volume), so the right is the left's mirror image (scmaps mirrors it on loading).
Each region's level-3 parent is kept for the cut-away views.

Usage: python tools/build_macaque_subcortex.py NMT_ROOT [--out src/scmaps/data/macaque]
       NMT_ROOT: the NMT v2.1-sym distribution (NMT_v2.1_sym/, tables_SARM/), e.g. from AFNI's @Install_NMT
Needs scipy and vtk besides numpy and nibabel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, diags

LEVELS = (3, 4, 5, 6)
OMIT = ("CbCx", "DCb", "wmCB", "Pit")    # level-3 regions not drawn (at finer levels, every region inside them)
GRID = 0.5                               # mm (the template's 0.25 mm coarsened 2x)
NETS_ITERATIONS = 16                     # surface nets' own smoothing (VTK's default)
SINC_ITERATIONS, PASSBAND = 40, 0.05
KEEP, MIN_FACES = 0.15, 200
CAP = 0.8                                # of the local half-thickness
MIDLINE_X = -0.05                        # mm: no left vertex beyond (the right is its mirror: the halves never touch)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_key(path):
    """[(index, abbreviation, full name)] of a SARM key (tab-separated, a header line)."""
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 3 and f[0].strip():
            rows.append((int(f[0]), f[1], f[2]))
    return rows


def coarse(vol):
    """The label volume on a 2x coarser grid: each 2x2x2 block its most frequent label, a tie to a region over 0."""
    sx, sy, sz = (s - s % 2 for s in vol.shape)
    v = vol[:sx, :sy, :sz]
    blocks = np.stack([v[i::2, j::2, k::2] for i in (0, 1) for j in (0, 1) for k in (0, 1)]).reshape(8, -1)
    count = (blocks[:, None, :] == blocks[None, :, :]).sum(1)
    best = blocks[(2 * count + (blocks > 0)).argmax(0), np.arange(blocks.shape[1])]
    return best.reshape(sx // 2, sy // 2, sz // 2)


def orient(F, V):
    """The faces turned consistently (an edge shared by two faces traversed once each way), then outward."""
    F = F.copy()
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    key = np.sort(e, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, fs = key[order], fid[order]
    same = (ks[1:] == ks[:-1]).all(1)
    adj = [[] for _ in range(len(F))]
    for x, y in zip(fs[:-1][same], fs[1:][same]):
        adj[x].append(y)
        adj[y].append(x)
    seen = np.zeros(len(F), bool)
    for s0 in range(len(F)):
        if seen[s0]:
            continue
        seen[s0] = True
        stack = [s0]
        while stack:
            f = stack.pop()
            ef = {(F[f, i], F[f, (i + 1) % 3]) for i in range(3)}
            for g in adj[f]:
                if not seen[g]:
                    if ef & {(F[g, i], F[g, (i + 1) % 3]) for i in range(3)}:
                        F[g] = F[g, ::-1]
                    seen[g] = True
                    stack.append(g)
    if (np.cross(V[F[:, 1]], V[F[:, 2]]) * V[F[:, 0]]).sum() < 0:
        F = F[:, ::-1]
    return F


def _poly(V, F):
    import vtk
    from vtk.util import numpy_support as ns
    p, q, c = vtk.vtkPolyData(), vtk.vtkPoints(), vtk.vtkCellArray()
    q.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, float), deep=True))
    p.SetPoints(q)
    c.ImportLegacyFormat(ns.numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].astype(np.int64).ravel(), deep=True))
    p.SetPolys(c)
    return p


def _arrays(poly):
    from vtk.util import numpy_support as ns
    V = ns.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    F = ns.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def surface_nets(vol, affine, labels):
    """{label: (vertices mm, faces)}: vtkSurfaceNets3D over the volume (`labels` only), each label's faces (those it shares
    with a neighbour or the background) turned outward."""
    import vtk
    from vtk.util import numpy_support as ns
    v = np.where(np.isin(vol, labels), vol, 0)
    nz = np.argwhere(v > 0)
    lo, hi = np.maximum(nz.min(0) - 2, 0), np.minimum(nz.max(0) + 3, v.shape)
    v = np.ascontiguousarray(v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    step = np.diag(affine)[:3]
    img = vtk.vtkImageData()
    img.SetDimensions(*v.shape)
    img.SetSpacing(*step)
    img.SetOrigin(*(affine[:3, 3] + lo * step))
    img.GetPointData().SetScalars(ns.numpy_to_vtk(v.ravel(order="F"), deep=True, array_type=vtk.VTK_INT))
    sn = vtk.vtkSurfaceNets3D()
    sn.SetInputData(img)
    for k, lab in enumerate(labels):
        sn.SetLabel(k, int(lab))
    sn.SetBackgroundLabel(0)
    sn.SetOutputMeshTypeToTriangles()
    sn.SmoothingOn()
    sn.SetNumberOfIterations(NETS_ITERATIONS)
    sn.Update()
    poly = sn.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return {}
    pts = ns.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    cells = ns.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    bl = ns.vtk_to_numpy(poly.GetCellData().GetArray("BoundaryLabels")).astype(np.int64)
    out = {}
    for lab in labels:
        F = np.concatenate([cells[bl[:, 0] == lab], cells[bl[:, 1] == lab]])
        if len(F) < 4:
            continue
        used = np.unique(F)
        remap = -np.ones(len(pts), np.int64)
        remap[used] = np.arange(len(used))
        V = pts[used]
        out[lab] = (V, orient(remap[F], V))
    return out


def simplify(V, F):
    """Windowed-sinc smoothing, then quadric decimation to KEEP of the triangles (at least MIN_FACES), turned outward."""
    import vtk
    sinc = vtk.vtkWindowedSincPolyDataFilter()
    sinc.SetInputData(_poly(V, F))
    sinc.SetNumberOfIterations(SINC_ITERATIONS)
    sinc.SetPassBand(PASSBAND)
    sinc.NonManifoldSmoothingOn()
    sinc.NormalizeCoordinatesOn()
    sinc.BoundarySmoothingOff()
    sinc.FeatureEdgeSmoothingOff()
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputConnection(sinc.GetOutputPort())
    dec.SetTargetReduction(1.0 - max(KEEP, min(1.0, MIN_FACES / max(len(F), 1))))
    dec.VolumePreservationOn()
    dec.Update()
    V2, F2 = _arrays(dec.GetOutput())
    return V2, orient(F2, V2)


def vertex_normals(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def pull_in(V, F, mask, affine, d, rounds=10):
    """Each vertex moved in along its normal by d mm, at most CAP of the local half-thickness: the region's inside distance
    (mm to its nearest outside voxel) at its first maximum along the inward normal, averaged over each vertex's neighbours
    `rounds` times (a noisy estimate would dent the surface)."""
    h = float(np.diag(affine)[0])
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = idx.max(0) + 4
    edt = ndimage.distance_transform_edt(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], sampling=h)
    n = vertex_normals(V, F)
    origin = affine[:3, 3] + lo * h
    half = np.zeros(len(V))
    alive = np.ones(len(V), bool)
    for step in np.arange(1, 17) * h / 2:
        val = ndimage.map_coordinates(edt, ((V - n * step - origin) / h).T, order=1, mode="constant", cval=0.0)
        alive &= ~(((val <= 0) & (half > 0)) | (val < 0.5 * half))
        half = np.where(alive, np.maximum(half, val), half)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V),) * 2).tocsr()
    A.data[:] = 1.0
    A = A + diags(np.ones(len(V)))
    A = diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(rounds):
        half = A @ half
    return V - n * np.minimum(d, CAP * half)[:, None]


def level_regions(vol, vol3, key, key3):
    """[(label, abbreviation, full name, level-3 parent)] of the level's left regions drawn, in the key's order."""
    name3 = {i: a for i, a, _ in key3}
    present = set(np.unique(vol[vol > 0]).tolist())
    out = []
    for lab, abbr, full in key:
        if not abbr.startswith("SL_") or lab not in present:
            continue
        parents = {name3[p][3:] for p in np.unique(vol3[vol == lab]) if p in name3}
        if parents & set(OMIT):
            continue
        if len(parents) != 1:
            raise AssertionError(f"{abbr}: not inside one level-3 region ({sorted(parents)})")
        out.append((lab, abbr, full, parents.pop()))
    return out


def build_level(vol, affine, regions):
    """{abbreviation: (vertices, faces)} and {abbreviation: grid mm} of the level's left regions."""
    grid = affine.copy()
    grid[:3, :3] *= 2
    grid[:3, 3] += 0.5 * np.diag(affine)[:3]
    coarse_vol = coarse(vol)
    labels = [r[0] for r in regions]
    nets = surface_nets(coarse_vol, grid, labels)
    out, used = {}, {}
    for lab, abbr, _full, _parent in regions:
        if lab in nets:
            g, aff, (V, F) = coarse_vol, grid, nets[lab]
        else:                                             # too small for the grid: alone on the template's
            alone = surface_nets(np.where(vol == lab, vol, 0), affine, [lab])
            if lab not in alone:
                raise AssertionError(f"{abbr}: no surface even on the 0.25 mm grid")
            g, aff, (V, F) = vol, affine, alone[lab]
        V, F = simplify(V, F)
        V = pull_in(V, F, g == lab, aff, 0.5 * float(np.diag(aff)[0]))
        V[:, 0] = np.minimum(V[:, 0], MIDLINE_X)          # smoothing can carry a midline region's rim across
        out[abbr], used[abbr] = (V, F), float(np.diag(aff)[0])
    return out, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nmt_root")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src" / "scmaps" / "data" / "macaque"))
    a = ap.parse_args()
    root = Path(a.nmt_root)
    sarm = root / "NMT_v2.1_sym" / "SARM_in_NMT_v2.1_sym.nii.gz"
    keys = {L: root / "tables_SARM" / f"SARM_key_{L}.txt" for L in LEVELS}
    img = nib.load(sarm)
    affine = img.affine
    if not np.allclose(affine[:3, :3], np.diag(np.full(3, 0.25))):
        raise SystemExit(f"{sarm}: expected an axis-aligned 0.25 mm grid")
    data = np.asanyarray(img.dataobj)
    vols = {L: np.asarray(data[:, :, :, 0, L - 1], np.int32) for L in LEVELS}
    key = {L: read_key(keys[L]) for L in LEVELS}
    x0 = affine[0, 3] + 0.25 * (vols[3].shape[0] - 1) / 2
    if abs(x0) > 1e-6:
        raise SystemExit(f"{sarm}: the midline is at x = {x0:g}, not 0")
    out = {}
    for L in LEVELS:                                      # symmetric: mirrored at x = 0, SL_ <-> SR_, the same volume
        ab = {abbr: i for i, abbr, _ in key[L]}
        lut = np.arange(vols[L].max() + 1)
        for abbr, i in ab.items():
            if abbr[:3] in ("SL_", "SR_") and (("SR_" if abbr[:3] == "SL_" else "SL_") + abbr[3:]) in ab:
                lut[i] = ab[("SR_" if abbr[:3] == "SL_" else "SL_") + abbr[3:]]
        if not np.array_equal(lut[vols[L][::-1]], vols[L]):
            raise SystemExit(f"SARM level {L} is not symmetric: the right cannot be the left's mirror")
        regions = level_regions(vols[L], vols[3], key[L], key[3])
        meshes, used = build_level(vols[L], affine, regions)
        names = [r[1] for r in regions]
        V = [meshes[n][0] for n in names]
        off = np.cumsum([0] + [len(v) for v in V])
        out[f"names_SARM{L}"] = np.array(names)
        out[f"full_SARM{L}"] = np.array([r[2] for r in regions])
        out[f"parent3_SARM{L}"] = np.array([r[3] for r in regions])
        out[f"grid_SARM{L}"] = np.array([used[n] for n in names], np.float32)
        out[f"level_SARM{L}"] = np.array([abbr for _i, abbr, _f in key[L]])
        out[f"points_SARM{L}"] = np.concatenate(V).astype(np.float32)
        out[f"faces_SARM{L}"] = np.concatenate([meshes[n][1] + o for n, o in zip(names, off[:-1])]).astype(np.int32)
        out[f"point_region_SARM{L}"] = np.repeat(np.arange(1, len(names) + 1), [len(v) for v in V]).astype(np.int16)
        finer = [n[3:] for n in names if used[n] < GRID]
        print(f"SARM{L}: {len(names)} regions a side, {len(out[f'faces_SARM{L}']):,} triangles"
              + (f"; on the 0.25 mm grid: {', '.join(finer)}" if finer else ""), flush=True)
    out["inputs_sha256"] = np.array(json.dumps({p.name: sha256(p) for p in (sarm, *keys.values())}, sort_keys=True))
    out["params"] = np.array(json.dumps(dict(grid_mm=GRID, nets_iterations=NETS_ITERATIONS, sinc_iterations=SINC_ITERATIONS,
                                             passband=PASSBAND, keep=KEEP, min_faces=MIN_FACES, cap=CAP,
                                             midline_x=MIDLINE_X, omit=OMIT)))
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "subcortex.npz", **out)
    print("->", dst / "subcortex.npz")


if __name__ == "__main__":
    main()
