"""The shipped atlas data (src/scmaps/data/, see its README for provenance):
the 32 Tian S2 subcortical ROI surfaces, and SUIT's cerebellar flatmap with
its underlay and the 32 Nettekoven Asym32 region labels."""

from functools import lru_cache
from importlib import resources

import numpy as np

# plotflatmap.m's underlay colour scale: gray(256) on [-1, 0.5]
UNDERSCALE = (-1.0, 0.5)
UNDERMAP_N = 256
MARGIN = 1.5                # voxels added to the larger half-extent (plotSubcortex.m)


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


@lru_cache(maxsize=1)
def flatmap():
    """SUIT's flatmap: the 2-D points, the triangles, the underlay grey per
    vertex as plotflatmap.m computes it (gray(256) indexed by
    ceil((d - lo) / (hi - lo) * 256), clipped to 1-256), and the Nettekoven
    label (1-32, current numbering) of every vertex."""
    import nibabel as nib
    with _path("FLAT.surf.gii").open("rb") as fh:
        pts, faces = nib.GiftiImage.from_bytes(fh.read()).agg_data()
    with _path("SUIT.shape.gii").open("rb") as fh:
        under = np.asarray(nib.GiftiImage.from_bytes(fh.read()).agg_data(), float)
    lo, hi = UNDERSCALE
    idx = np.clip(np.ceil((under - lo) / (hi - lo) * UNDERMAP_N), 1, UNDERMAP_N)
    grey = (idx - 1) / (UNDERMAP_N - 1)
    lab = _cerebellum_npz()["labels"].astype(int)
    if not (len(lab) == len(pts) == len(grey)) or lab.min() < 1 or lab.max() > 32:
        raise AssertionError("flatmap geometry, underlay and labels do not line up")
    return np.asarray(pts, float)[:, :2], np.asarray(faces, np.int64), grey, lab
