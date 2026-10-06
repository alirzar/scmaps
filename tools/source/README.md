# Source files of the shipped atlas data

`tools/convert_resources.py` turns these into the `.npz` files in
`src/scmaps/data/`, checking each file's sha256 first. They are kept here as
the provenance record; the package itself does not read them. Licences and
citations: `NOTICE` at the repository root.

| file | what it is | from |
|---|---|---|
| `subcorticalSurf.mat` | the 32 Tian S2 ROI surfaces: `VL` (left hemisphere, plotting positions 1-16) and `VR` (right, 17-32), each the triangles of MATLAB's `boundary(..., 0.5)` over the ROI's voxels of the Tian S2 3T atlas (MNI 2 mm) | the scmaps author's MATLAB plotting code (`makeSurface.m`, run once) |
| `subcorticalMapping.csv` | Tian ROI name -> plotting position (`roi_ix_LtoR`: left 1-16, right 17-32) | the same code |
| `atl-NettekovenAsym32_dseg.label.gii` | the Nettekoven asymmetric 32-region label of each of the 28,935 SUIT flatmap vertices (in the atlas's current numbering), with the region names | [DiedrichsenLab/cerebellar_atlases](https://github.com/DiedrichsenLab/cerebellar_atlases), `Nettekoven_2024/` |
| `atl-NettekovenAsym32.lut` | the atlas's label -> colour and name table | the same folder |

sha256:

    3594eb04f8c24ec572b546808ad1f64ffa084bfda8767f900957dfcdb6572342  subcorticalSurf.mat
    e02d7c8edd9ed9b328081d3772ecab120ebecc24a91f2f755871cb159a010c1d  subcorticalMapping.csv
    275c332a4615e8e99c48a37142c7a5fc3bcd12f01fd8507fccd518937c8a1828  atl-NettekovenAsym32_dseg.label.gii
    33a723b32cb861623a8459865454c9efb142bdb3ce2390ea5f29a586cfab0e24  atl-NettekovenAsym32.lut
