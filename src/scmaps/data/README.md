# Atlas data

Everything the renderers read. See `NOTICE` at the repository root for
citations and licences.

| file | what it is | from |
|---|---|---|
| `subcortex_surfaces.npz` | the 32 Tian S2 (3T, MNI 2 mm) ROI surfaces: every ROI's voxel points (`points`, `point_roi` 1-32), the 0-based triangles of MATLAB's `boundary(..., 0.5)` over them (`faces`, `face_roi`), and the ROI names in plotting order (`names`: left 1-16, right 17-32) | `tools/source/subcorticalSurf.mat` + `subcorticalMapping.csv`, converted by `tools/convert_resources.py` |
| `cerebellum_labels.npz` | the Nettekoven Asym32 label (1-32 in the atlas's current numbering; 1-16 left, 17-32 right) of each of the 28,935 SUIT flatmap vertices (`labels`), and the names of labels 1-32 (`names`: M1L ... S5L, M1R ... S5R) | `tools/source/atl-NettekovenAsym32_dseg.label.gii`, the atlas's own file, converted by the same script |
| `FLAT.surf.gii` | SUIT's cerebellar flatmap surface | SUITPy, `SUITPy/surfaces/FLAT.surf.gii` |
| `SUIT.shape.gii` | the flatmap's shading underlay | SUITPy, `SUITPy/surfaces/SUIT.shape.gii` |
| `macaque/FLAT.surf.gii` | the macaque cerebellar flatmap: 23,772 vertices on a 0.25 mm grid of the unfolded cerebellar cortex, both hemispheres, in flat units spanning about ±90 like SUIT's (`mm_per_unit` in the labels file) | `tools/build_macaque_flatmap.py`, from AFNI's NMT v2.1-sym (its tissue segmentation and SARM) |
| `macaque/NMT.shape.gii` | its shading underlay: the depth below the cerebellar envelope, with SUIT.shape.gii's value distribution | the same script |
| `macaque/subcortex.npz` | SARM's subcortical regions at levels 3-6 (all but the cerebellum's and the pituitary), the left hemisphere's surfaces: per level its points (`points_SARM*`, mm in NMT v2.1-sym), triangles (`faces_SARM*`), each point's region (`point_region_SARM*`, 1..n), the regions' names, full names and level-3 parents (`names_SARM*`, `full_SARM*`, `parent3_SARM*`), the grid each was built on (`grid_SARM*`), every name of the level (`level_SARM*`), the build's parameters (`params`) and the sha256 of the NMT inputs (`inputs_sha256`); the right hemisphere is the left mirrored at x = 0 | `tools/build_macaque_subcortex.py`, from AFNI's NMT v2.1-sym (its SARM) |
| `macaque/cerebellum_labels.npz` | each vertex's SARM cerebellar cortex region at levels 3-6 (`labels_SARM3` ... `labels_SARM6`, 1..n), the regions' names (`names_SARM*`: SL_ left, then SR_ right) and full names (`full_SARM*`), `mm_per_unit`, and the sha256 of the NMT inputs (`inputs_sha256`) | the same script |

The conversion is lossless: the meshes and the frame built from the `.npz`
files equal, array for array, those built from the `.mat` file, and the labels
and names equal the atlas file's.

sha256:

    d8488577913047da8b0a47c2756f69e9383a4a476c4afc5d19f1f78e3d379e0d  FLAT.surf.gii
    e396a6d518d05aead246b80fc3dc90b6f4d154d500297090040295c4e9c3ab2c  SUIT.shape.gii
    6d47d2b91ab9eae7f93070a4b067d4ec7869e7ddd3d48d4f475de95ad86c717d  macaque/cerebellum_labels.npz
    4b23e2b6fe11b76e1401381da0ce1db8814f171533bae7fbd1a992c6c55abddd  macaque/FLAT.surf.gii
    9d1bbafcfacb430a38b28f7891b36f93f800655303ee939061f548404cb70a38  macaque/NMT.shape.gii
    d48e02159c52d51bf849a34a2cca4dbcf7003aa63de735838714566295eb6d6d  macaque/subcortex.npz

Rebuild the macaque files with `python tools/build_macaque_flatmap.py NMT_ROOT`
and `python tools/build_macaque_subcortex.py NMT_ROOT` (NMT_ROOT: the NMT
v2.1-sym distribution, with `NMT_v2.1_sym/` and `tables_SARM/`); they need
scipy besides the package's own dependencies.
