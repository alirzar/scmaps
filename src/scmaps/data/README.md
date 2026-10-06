# Atlas data

Everything the renderers read. See `NOTICE` at the repository root for
citations and licences.

| file | what it is | from |
|---|---|---|
| `subcortex_surfaces.npz` | the 32 Tian S2 (3T, MNI 2 mm) ROI surfaces: every ROI's voxel points (`points`, `point_roi` 1-32), the 0-based triangles of MATLAB's `boundary(..., 0.5)` over them (`faces`, `face_roi`), and the ROI names in plotting order (`names`: left 1-16, right 17-32) | `tools/source/subcorticalSurf.mat` + `subcorticalMapping.csv`, converted by `tools/convert_resources.py` |
| `cerebellum_labels.npz` | the Nettekoven Asym32 label (1-32 in the atlas's current numbering; 1-16 left, 17-32 right) of each of the 28,935 SUIT flatmap vertices (`labels`), and the names of labels 1-32 (`names`: M1L ... S5L, M1R ... S5R) | `tools/source/atl-NettekovenAsym32_dseg.label.gii`, the atlas's own file, converted by the same script |
| `FLAT.surf.gii` | SUIT's cerebellar flatmap surface | SUITPy, `SUITPy/surfaces/FLAT.surf.gii` |
| `SUIT.shape.gii` | the flatmap's shading underlay | SUITPy, `SUITPy/surfaces/SUIT.shape.gii` |

The conversion is lossless: the meshes and the frame built from the `.npz`
files equal, array for array, those built from the `.mat` file, and the labels
and names equal the atlas file's.

sha256:

    d8488577913047da8b0a47c2756f69e9383a4a476c4afc5d19f1f78e3d379e0d  FLAT.surf.gii
    e396a6d518d05aead246b80fc3dc90b6f4d154d500297090040295c4e9c3ab2c  SUIT.shape.gii
