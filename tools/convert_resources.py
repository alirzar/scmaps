"""Convert the atlas source files in tools/source/ to the plain numpy files the
package ships (src/scmaps/data/), and check the conversion is lossless.

Run once, by hand, when a source file changes:

    python tools/convert_resources.py

Needs scipy (to read the .mat file); the package itself does not.

* subcorticalSurf.mat (VL / VR: per ROI the voxel points x, y, z and the
  1-based triangles S of MATLAB's boundary(..., 0.5)) ->
  subcortex_surfaces.npz: every ROI's points stacked in plotting order
  (`points`, `point_roi` 1-32), its triangles as 0-based indices into
  `points` (`faces`, `face_roi`), and the 32 Tian S2 names in plotting order
  (`names`, from subcorticalMapping.csv's roi_ix_LtoR).
* atl-NettekovenAsym32_dseg.label.gii (the Nettekoven asymmetric 32-region
  label of each SUIT flatmap vertex, in the atlas's current numbering) ->
  cerebellum_labels.npz (`labels` 1-32, `names` of labels 1-32).
"""

import hashlib
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "source")
OUT = os.path.join(ROOT, "src", "scmaps", "data")

SHA256 = {
    "subcorticalSurf.mat": "3594eb04f8c24ec572b546808ad1f64ffa084bfda8767f900957dfcdb6572342",
    "subcorticalMapping.csv": "e02d7c8edd9ed9b328081d3772ecab120ebecc24a91f2f755871cb159a010c1d",
    "atl-NettekovenAsym32_dseg.label.gii":
        "275c332a4615e8e99c48a37142c7a5fc3bcd12f01fd8507fccd518937c8a1828",
    "atl-NettekovenAsym32.lut": "33a723b32cb861623a8459865454c9efb142bdb3ce2390ea5f29a586cfab0e24",
}
# the atlas's current numbering
NETTEKOVEN = tuple(f"{r}{h}" for h in "LR" for r in (
    "M1", "M2", "M3", "M4", "A1", "A2", "A3", "D1", "D2", "D3", "D4", "S1", "S2", "S3", "S4", "S5"))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _points(v):
    return np.c_[np.atleast_1d(v.x), np.atleast_1d(v.y), np.atleast_1d(v.z)].astype(float)


def convert_subcortex():
    s = sio.loadmat(os.path.join(SRC, "subcorticalSurf.mat"), squeeze_me=True,
                    struct_as_record=False)
    m = pd.read_csv(os.path.join(SRC, "subcorticalMapping.csv")).sort_values("roi_ix_LtoR")
    names = m["regions"].to_numpy(str)
    assert list(m["roi_ix_LtoR"]) == list(range(1, 33))
    assert all(n.endswith("-lh") for n in names[:16]) and all(n.endswith("-rh") for n in names[16:])
    rois = list(s["VL"]) + list(s["VR"])                 # plotting positions 1-16, 17-32
    points, point_roi, faces, face_roi, n0 = [], [], [], [], 0
    for p, v in enumerate(rois, start=1):
        P = _points(v)
        F = np.atleast_2d(np.asarray(v.S, int)) - 1      # MATLAB is 1-based
        points.append(P)
        point_roi.append(np.full(len(P), p))
        faces.append(F + n0)
        face_roi.append(np.full(len(F), p))
        n0 += len(P)
    out = dict(points=np.vstack(points), point_roi=np.concatenate(point_roi).astype(np.int8),
               faces=np.vstack(faces).astype(np.int32),
               face_roi=np.concatenate(face_roi).astype(np.int8), names=names)
    np.savez_compressed(os.path.join(OUT, "subcortex_surfaces.npz"), **out)
    return s


def convert_cerebellum():
    g = nib.load(os.path.join(SRC, "atl-NettekovenAsym32_dseg.label.gii"))
    lab = np.asarray(g.agg_data()).astype(int).ravel()
    table = g.labeltable.get_labels_as_dict()
    names = tuple(table[k] for k in range(1, 33))
    with open(os.path.join(SRC, "atl-NettekovenAsym32.lut")) as fh:
        lut = {int(r.split()[0]): r.split()[-1] for r in fh if r.strip()}
    assert names == NETTEKOVEN == tuple(lut[k] for k in range(1, 33)), names
    assert lab.min() == 1 and lab.max() == 32 and len(np.unique(lab)) == 32
    np.savez_compressed(os.path.join(OUT, "cerebellum_labels.npz"), labels=lab.astype(np.int8),
                        names=np.asarray(names))
    return lab, names


def _matlab_meshes(s):
    """The meshes and the frame built straight from the .mat, the way the
    original renderer built them: what the .npz files must reproduce."""
    out, ctr, half = {}, {}, np.zeros(3)
    for hemi, key, off in (("L", "VL", 0), ("R", "VR", 16)):
        pts, faces, roi, n0 = [], [], [], 0
        for p, v in enumerate(s[key]):
            P = _points(v)
            F = np.atleast_2d(np.asarray(v.S, int)) - 1
            used = np.unique(F)
            remap = np.full(len(P), -1)
            remap[used] = np.arange(len(used))
            pts.append(P[used])
            faces.append(remap[F] + n0)
            roi.append(np.full(len(used), p + 1 + off))
            n0 += len(used)
        out[hemi] = (np.vstack(pts), np.vstack(faces), np.concatenate(roi))
        allp = np.vstack([_points(v) for v in s[key]])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ctr[hemi] = (lo + hi) / 2
        half = np.maximum(half, (hi - lo) / 2)
    return out, ctr, half + 1.5


def main():
    for f, want in SHA256.items():
        got = _sha(os.path.join(SRC, f))
        if got != want:
            sys.exit(f"{f}: sha256 {got} is not the recorded {want}")
    s = convert_subcortex()
    lab, names = convert_cerebellum()

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from scmaps import _atlas
    for fn in (_atlas.subcortex_meshes, _atlas.subcortex_frame, _atlas.flatmap,
               _atlas.cerebellum_names):
        fn.cache_clear()
    want, ctr, half = _matlab_meshes(s)
    got = _atlas.subcortex_meshes()
    for hemi in "LR":
        for a, b in zip(want[hemi], got[hemi]):
            np.testing.assert_array_equal(a, b)
    gctr, ghalf = _atlas.subcortex_frame()
    np.testing.assert_array_equal(half, ghalf)
    for hemi in "LR":
        np.testing.assert_array_equal(ctr[hemi], gctr[hemi])
    np.testing.assert_array_equal(_atlas.flatmap()[3], lab)
    assert _atlas.cerebellum_names() == names
    print("converted; meshes and frame identical to the .mat source, labels and names to the "
          "atlas file")
    for f in ("subcortex_surfaces.npz", "cerebellum_labels.npz"):
        print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.0f} kB")


if __name__ == "__main__":
    main()
