"""The shipped atlas data (src/scmaps/data/, see its README for provenance):
the 32 Tian S2 subcortical ROI surfaces, SUIT's cerebellar flatmap with its
underlay and the 32 Nettekoven Asym32 region labels; and, from NMT v2.1-sym
and its SARM atlas (data/macaque/), the macaque cerebellar flatmap with its
underlay and SARM's cerebellar regions at levels 3 to 6, and the surfaces of
SARM's other subcortical regions at those levels."""

from functools import lru_cache
from importlib import resources

import numpy as np

# plotflatmap.m's underlay colour scale: gray(256) on [-1, 0.5]
UNDERSCALE = (-1.0, 0.5)
UNDERMAP_N = 256
MARGIN = 1.5                # voxels added to the larger half-extent (plotSubcortex.m)
MACAQUE_MARGIN = 1.0        # mm added to the macaque subcortex's larger half-extent


def _path(name):
    return resources.files("scmaps") / "data" / name


@lru_cache(maxsize=1)
def _subcortex_npz():
    with _path("subcortex_surfaces.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def subcortex_names():
    """The 32 Tian S2 ROI names in plotting order: left hemisphere 1-16, then
    the right hemisphere's same 16 structures."""
    return tuple(str(n) for n in _subcortex_npz()["names"])


@lru_cache(maxsize=1)
def subcortex_meshes():
    """{"L"/"R": (points, faces, ROI position 1-32 per point)}: each
    hemisphere's 16 ROI surfaces as one mesh, positions 1-16 left, 17-32
    right. Only the points the boundary faces use are kept (the source keeps
    every voxel of the ROI, interior ones too)."""
    z = _subcortex_npz()
    P, proi, F, froi = z["points"], z["point_roi"], z["faces"].astype(np.int64), z["face_roi"]
    out = {}
    for hemi, rois in (("L", range(1, 17)), ("R", range(17, 33))):
        pts, faces, roi, n0 = [], [], [], 0
        for p in rois:
            first = np.flatnonzero(proi == p)[0]
            f = F[froi == p] - first                     # local to the ROI's points
            used = np.unique(f)
            remap = np.full(int((proi == p).sum()), -1)
            remap[used] = np.arange(len(used))
            pts.append(P[first + used])
            faces.append(remap[f] + n0)
            roi.append(np.full(len(used), p))
            n0 += len(used)
        out[hemi] = (np.vstack(pts), np.vstack(faces), np.concatenate(roi))
    return out


@lru_cache(maxsize=1)
def subcortex_frame():
    """({"L"/"R": box centre}, half-extents): plotSubcortex.m's one scale --
    each hemisphere centred in its own box, every box with the larger
    half-extent of the two hemispheres plus MARGIN, computed over ALL of each
    ROI's points (interior voxels included), as MATLAB computes it."""
    z = _subcortex_npz()
    ctr, half = {}, np.zeros(3)
    for hemi, sel in (("L", z["point_roi"] <= 16), ("R", z["point_roi"] > 16)):
        P = z["points"][sel]
        lo, hi = P.min(axis=0), P.max(axis=0)
        ctr[hemi] = (lo + hi) / 2
        half = np.maximum(half, (hi - lo) / 2)
    return ctr, half + MARGIN


@lru_cache(maxsize=1)
def _cerebellum_npz():
    with _path("cerebellum_labels.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def cerebellum_names():
    """The names of the Nettekoven Asym32 labels 1-32, in the atlas's current
    numbering: M1L ... S5L (left), then M1R ... S5R (right)."""
    return tuple(str(n) for n in _cerebellum_npz()["names"])


def _flat(surf, shape):
    """A flatmap's 2-D points, triangles and the underlay grey per vertex as
    plotflatmap.m computes it (gray(256) indexed by ceil((d - lo) / (hi - lo)
    * 256), clipped to 1-256)."""
    import nibabel as nib
    with _path(surf).open("rb") as fh:
        pts, faces = nib.GiftiImage.from_bytes(fh.read()).agg_data()
    with _path(shape).open("rb") as fh:
        under = np.asarray(nib.GiftiImage.from_bytes(fh.read()).agg_data(), float)
    lo, hi = UNDERSCALE
    idx = np.clip(np.ceil((under - lo) / (hi - lo) * UNDERMAP_N), 1, UNDERMAP_N)
    return np.asarray(pts, float)[:, :2], np.asarray(faces, np.int64), (idx - 1) / (UNDERMAP_N - 1)


@lru_cache(maxsize=1)
def flatmap():
    """SUIT's flatmap: the 2-D points, the triangles, the underlay grey per
    vertex, and the Nettekoven label (1-32, current numbering) of every
    vertex."""
    pts, faces, grey = _flat("FLAT.surf.gii", "SUIT.shape.gii")
    lab = _cerebellum_npz()["labels"].astype(int)
    if not (len(lab) == len(pts) == len(grey)) or lab.min() < 1 or lab.max() > 32:
        raise AssertionError("flatmap geometry, underlay and labels do not line up")
    return pts, faces, grey, lab


MACAQUE_ATLASES = ("SARM3", "SARM4", "SARM5", "SARM6")   # SARM's cerebellar levels on the macaque flatmap


@lru_cache(maxsize=1)
def _macaque_npz():
    with _path("macaque/cerebellum_labels.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: z[k] for k in z.files}


def _check_macaque(atlas):
    if atlas not in MACAQUE_ATLASES:
        raise ValueError(f"unknown macaque cerebellar atlas {atlas!r}; one of {MACAQUE_ATLASES}")


@lru_cache(maxsize=None)
def macaque_cerebellum_names(atlas):
    """(names, full names) of a SARM level's cerebellar cortex regions on the
    macaque flatmap: the left hemisphere's (SL_), then the right's (SR_), e.g.
    "SL_Cb1V" ... "SR_Fl" for SARM6."""
    _check_macaque(atlas)
    z = _macaque_npz()
    return tuple(str(n) for n in z[f"names_{atlas}"]), tuple(str(n) for n in z[f"full_{atlas}"])


@lru_cache(maxsize=None)
def macaque_flatmap(atlas):
    """The macaque cerebellar flatmap (NMT v2.1-sym, tools/build_macaque_flatmap.py):
    the 2-D points, the triangles, the underlay grey per vertex, and the label
    of every vertex (1..n in `macaque_cerebellum_names(atlas)` order)."""
    _check_macaque(atlas)
    pts, faces, grey = _flat("macaque/FLAT.surf.gii", "macaque/NMT.shape.gii")
    lab = _macaque_npz()[f"labels_{atlas}"].astype(int)
    n = len(macaque_cerebellum_names(atlas)[0])
    if not (len(lab) == len(pts) == len(grey)) or lab.min() < 1 or lab.max() > n:
        raise AssertionError("macaque flatmap geometry, underlay and labels do not line up")
    return pts, faces, grey, lab


def macaque_mm_per_unit():
    """Millimetres of the template's cerebellar surface per macaque flatmap unit."""
    return float(_macaque_npz()["mm_per_unit"])


@lru_cache(maxsize=1)
def _macaque_subcortex_npz():
    with _path("macaque/subcortex.npz").open("rb") as fh:
        z = np.load(fh)
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def macaque_subcortex_names(atlas):
    """(names, full names, level-3 parents) of a SARM level's subcortical
    regions as drawn (tools/build_macaque_subcortex.py: every region but the
    cerebellum's and the pituitary): the left hemisphere's (SL_) in SARM's key
    order, then the same regions of the right (SR_)."""
    _check_macaque(atlas)
    z = _macaque_subcortex_npz()
    left = [str(n) for n in z[f"names_{atlas}"]]
    full = [str(n) for n in z[f"full_{atlas}"]]
    parents = [str(n) for n in z[f"parent3_{atlas}"]]
    return (tuple(left + ["SR_" + n[3:] for n in left]), tuple(full + ["SR_" + n[3:] for n in full]),
            tuple(parents + parents))


@lru_cache(maxsize=None)
def sarm_names(atlas):
    """Every region name of a SARM level, both hemispheres, the regions no
    scmaps map draws (the cerebellar nuclei and white matter, the pituitary)
    included."""
    _check_macaque(atlas)
    return tuple(str(n) for n in _macaque_subcortex_npz()[f"level_{atlas}"])


@lru_cache(maxsize=None)
def macaque_subcortex_meshes(atlas):
    """{"L"/"R": (points, faces, region position per point)}: a SARM level's
    subcortical surfaces, positions 1..n on the left, n+1..2n on the right;
    the right is the left mirrored at x = 0 (SARM in NMT v2.1-sym is
    symmetric), its faces turned to stay outward."""
    _check_macaque(atlas)
    z = _macaque_subcortex_npz()
    P = z[f"points_{atlas}"].astype(float)
    F = z[f"faces_{atlas}"].astype(np.int64)
    roi = z[f"point_region_{atlas}"].astype(np.int64)
    n = len(z[f"names_{atlas}"])
    if roi.min() != 1 or roi.max() != n or F.min() != 0 or F.max() != len(P) - 1:
        raise AssertionError(f"{atlas}: the subcortical surfaces and their regions do not line up")
    return {"L": (P, F, roi), "R": (P * [-1.0, 1.0, 1.0], F[:, ::-1].copy(), roi + n)}


@lru_cache(maxsize=None)
def macaque_subcortex_frame(atlas):
    """({"L"/"R": box centre}, half-extents) as subcortex_frame computes the
    human one: each hemisphere centred in its own box, one box size (the
    surfaces' half-extent plus MACAQUE_MARGIN mm)."""
    ctr, half = {}, np.zeros(3)
    for hemi, (P, _faces, _roi) in macaque_subcortex_meshes(atlas).items():
        lo, hi = P.min(axis=0), P.max(axis=0)
        ctr[hemi] = (lo + hi) / 2
        half = np.maximum(half, (hi - lo) / 2)
    return ctr, half + MACAQUE_MARGIN
