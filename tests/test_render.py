"""The renderers: MATLAB's camera, light and material constants as measured on
plotSubcortex.m's own figures, plotflatmap's colour arithmetic, the layouts,
and -- against two renders the MATLAB code made (tests/data) -- that the port
draws the same images, at thresholds every deliberately broken port tried
fails (two ROIs swapped, the hemispheres swapped, the light on the wrong
side, the views mirrored, the ambient or the flatmap opacity changed, the seed
shaded)."""

import json
import pathlib

import numpy as np
import pytest

import scmaps
from scmaps import _atlas, cerebellum as C, subcortex as S

REFERENCES = pathlib.Path(__file__).parent / "data"


def test_atlas_lines_up():
    names = _atlas.subcortex_names()
    assert len(names) == 32 and names[0] == "aHIP-lh" and names[16] == "aHIP-rh"
    assert all(n.endswith("-lh") for n in names[:16]) and all(n.endswith("-rh") for n in names[16:])
    assert [n[:-3] for n in names[:16]] == [n[:-3] for n in names[16:]]
    meshes = _atlas.subcortex_meshes()
    for hemi, rois in (("L", range(1, 17)), ("R", range(17, 33))):
        pts, faces, roi = meshes[hemi]
        assert sorted(np.unique(roi).tolist()) == list(rois)
        assert faces.min() == 0 and faces.max() == len(pts) - 1
    # the left hemisphere lies at smaller x than the right
    assert meshes["L"][0][:, 0].mean() < meshes["R"][0][:, 0].mean()
    pts, _faces, _grey, lab = _atlas.flatmap()
    assert sorted(np.unique(lab).tolist()) == list(range(1, 33))
    assert pts[lab <= 16, 0].mean() < 0 < pts[lab > 16, 0].mean()   # 1-16 left
    cer = _atlas.cerebellum_names()                 # the atlas's current numbering
    assert cer[:8] == ("M1L", "M2L", "M3L", "M4L", "A1L", "A2L", "A3L", "D1L")
    assert cer[15] == "S5L" and cer[16:] == tuple(n[:-1] + "R" for n in cer[:16])


def test_original_numbering_moves_only_s5():
    """The atlas's original release numbered today's S5 as A4, label 8 (24),
    and D1-S4 as 9-16 (25-32); every other label is unchanged."""
    cur, orig = scmaps.CEREBELLUM_REGIONS, scmaps.CEREBELLUM_REGIONS_ORIGINAL
    assert sorted(orig) == sorted(cur)
    moved = [k for k in range(32) if cur[k] != orig[k]]
    assert [k + 1 for k in moved] == [*range(8, 17), *range(24, 33)]
    assert (orig[7], orig[23]) == ("S5L", "S5R")
    assert orig[8:16] == cur[7:15] and orig[24:32] == cur[23:31]


def test_vertex_normals_are_matlabs_unit_face_normal_sum():
    """MATLAB's patch VertexNormals sum the faces' UNIT normals (measured in
    R2024b to 2e-6 deg): a big and a small face meeting at a vertex count the
    same, which an area-weighted sum would not."""
    pts = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0.5, 0.5]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    n = S.vertex_normals(pts, faces)
    f1 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    f2 = np.cross(pts[2] - pts[0], pts[3] - pts[0])
    unit = f1 / np.linalg.norm(f1) + f2 / np.linalg.norm(f2)
    np.testing.assert_allclose(n[0], unit / np.linalg.norm(unit), atol=1e-12)
    area = (f1 + f2) / np.linalg.norm(f1 + f2)
    assert np.degrees(np.arccos(np.clip(n[0] @ area, -1, 1))) > 10


def test_camera_light_and_material_are_what_matlab_used():
    """Measured on the figure plotSubcortex.m made in MATLAB R2024b (left
    hemisphere, view(-90, 0)): CameraTarget, CameraViewAngle, the camlight's
    position; material dull; the grey of an ROI without a value."""
    ctr, half = _atlas.subcortex_frame()
    np.testing.assert_allclose(ctr["L"], [-54.5, 59.5, 35.5])
    d = S.CAM_DIST * np.linalg.norm(half)
    assert np.degrees(2 * np.arctan(half[2] / d)) == pytest.approx(6.436174, abs=1e-6)
    np.testing.assert_allclose(S.light_position(ctr["L"], -1),
                               [-241.248494, -48.319293, 159.998996], atol=1e-5)
    assert (S.AMBIENT, S.DIFFUSE, S.GREY, S.TILE_IN) == (0.3, 0.8, 0.82, 1.2)


def test_lighting_is_dull_material_and_reverselit():
    """A square facing the camera at the box centre: colour x (0.3 + 0.8 cos),
    cos to the local light; reversing the faces' winding changes nothing."""
    ctr, _half = _atlas.subcortex_frame()
    c = ctr["L"]
    sq = c + np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], float) * 1e-3
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    rgb = np.tile([0.2, 0.6, 1.0], (4, 1))
    lit = S.lit_colours(sq, faces, rgb, c, -1)
    L = S.light_position(c, -1) - sq
    cos = (L @ [-1.0, 0, 0]) / np.linalg.norm(L, axis=1)
    np.testing.assert_allclose(lit, rgb * (0.3 + 0.8 * cos)[:, None], atol=1e-9)
    np.testing.assert_allclose(S.lit_colours(sq, faces[:, ::-1], rgb, c, -1), lit, atol=1e-12)


def test_flatmap_colours_follow_plotflatmap():
    """plotflatmap: overlay .* ((1 - a) + a .* underlay); no value shows the
    underlay; the underlay is gray(256) indexed ceil((d + 1) / 1.5 * 256)."""
    _pts, _faces, grey, lab = _atlas.flatmap()
    assert grey.min() >= 0 and grey.max() <= 1
    np.testing.assert_allclose(grey * 255, np.round(grey * 255), atol=1e-9)   # gray(256) levels
    rgb = np.tile([0.9, 0.5, 0.1], (32, 1))
    rgb[7] = np.nan
    a = np.full(32, 0.75)
    a[3] = 0.0
    col = C.flat_colours(rgb, a)
    k = lab - 1
    g = grey[:, None]
    np.testing.assert_allclose(col[k == 7], np.repeat(g[k == 7], 3, axis=1))      # the underlay
    np.testing.assert_allclose(col[k == 3], np.tile([0.9, 0.5, 0.1], ((k == 3).sum(), 1)))  # seed
    other = (k != 7) & (k != 3)
    np.testing.assert_allclose(col[other], [0.9, 0.5, 0.1] * (0.25 + 0.75 * g[other]))


def test_every_layout_renders_rgba_on_a_transparent_background():
    rgb = np.tile([0.3, 0.5, 0.7], (32, 1))
    shapes = {}
    for layout in S.LAYOUTS:
        im = S.render_subcortex(rgb, layout, dpi=60)
        assert im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 4
        assert (im[..., 3] == 0).any() and (im[..., 3] == 255).any()
        shapes[layout] = im.shape[:2]
    # tiles edge to edge at one scale: row = 2 x lateral = 4 x left wide; grid = 2 x lateral tall
    assert shapes["row"][1] == pytest.approx(2 * shapes["lateral"][1], rel=0.1)
    assert shapes["lateral"][1] == pytest.approx(2 * shapes["left"][1], rel=0.15)
    assert shapes["grid"][0] == pytest.approx(2 * shapes["lateral"][0], rel=0.15)
    flat = C.render_cerebellum(rgb, 0.75, dpi=60)
    assert (flat[..., 3] == 0).any() and (flat[..., 3] == 255).any()


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError):
        S.render_subcortex(np.zeros((32, 3)), "flat", 60)


def test_no_value_is_grey():
    nan = np.full((32, 3), np.nan)
    for im in (S.render_subcortex(nan, "left", dpi=60), C.render_cerebellum(nan, 0.75, dpi=60)):
        px = im[im[..., 3] == 255][:, :3].astype(int)
        assert len(px) and (px.max(axis=1) - px.min(axis=1)).max() <= 1


# --- the port against the MATLAB renders ------------------------------------

def _aligned(m, p, reach=2):
    """Both on one canvas, the Python image shifted by the whole-pixel offset
    (within +-reach) that best overlaps the silhouettes."""
    H, W = max(m.shape[0], p.shape[0]) + 2 * reach, max(m.shape[1], p.shape[1]) + 2 * reach

    def put(x):
        out = np.zeros((H, W, 4), np.uint8)
        out[reach:reach + x.shape[0], reach:reach + x.shape[1]] = x
        return out
    M, P0 = put(m), put(p)
    best = None
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            P = np.roll(P0, (dy, dx), axis=(0, 1))
            am, ap = M[..., 3] > 127, P[..., 3] > 127
            iou = (am & ap).sum() / max((am | ap).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, P)
    return M, best[1], best[0]


def _blurred(x, sigma=1.0):
    """Colour after a sigma-px blur in premultiplied space: MATLAB's and VTK's
    rasterisers sit ~0.3 px apart, which the blur absorbs; a wrong colour on a
    whole ROI survives it."""
    from scipy.ndimage import gaussian_filter
    f = x.astype(float) / 255
    a = gaussian_filter(f[..., 3], sigma)
    rgb = np.stack([gaussian_filter(f[..., c] * f[..., 3], sigma) for c in range(3)], axis=-1)
    return rgb / np.maximum(a, 1e-9)[..., None], a


def _to_current(x):
    """Per-region rows in the atlas's original numbering -> current numbering
    (the MATLAB code numbered the cerebellar regions as the original release
    did), matched by name."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    out[[scmaps.CEREBELLUM_REGIONS.index(n) for n in scmaps.CEREBELLUM_REGIONS_ORIGINAL]] = x
    return out


def _against_matlab(kind, reorder=True):
    """(silhouette IoU, per-pixel colour differences inside both) of the port
    against the MATLAB render of the same map."""
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("scipy")
    ref = json.loads((REFERENCES / "references.json").read_text())
    mp = next(m for m in ref["maps"] if m["kind"] == kind)
    rgb = np.array([[np.nan if c is None else c for c in row] for row in mp["rgb"]], float)
    matlab = np.asarray(Image.open(REFERENCES / mp["file"]).convert("RGBA"))
    if kind == "subcortex":
        python = S.render_subcortex(rgb, mp["layout"], mp["dpi"])
    else:
        alphas = np.asarray(mp["alphas"], float)
        if reorder:
            rgb, alphas = _to_current(rgb), _to_current(alphas)
        python = C.render_cerebellum(rgb, alphas, mp["dpi"])
    M, P, iou = _aligned(matlab, python)
    cm, am = _blurred(M)
    cp, ap = _blurred(P)
    inside = (am > 0.98) & (ap > 0.98)
    return iou, np.abs(cm - cp).mean(axis=2)[inside]


@pytest.mark.parametrize("kind, min_iou", (("subcortex", 0.97), ("cerebellum", 0.99)))
def test_the_port_draws_what_matlab_drew(kind, min_iou):
    """At 150 dpi the port scored IoU 0.983 / 0.995, mean colour difference
    0.007 / 0.009 and 0.5% / 0.0% of pixels off by more than 0.1; the mildest
    broken port tried (two small ROIs swapped) put 1.8% of pixels off, the
    others >= 4.4% or a mean >= 0.024."""
    iou, d = _against_matlab(kind)
    assert iou >= min_iou
    assert d.mean() <= 0.015
    assert (d > 0.1).mean() <= 0.01


def test_the_cerebellar_numbering_matters():
    """The MATLAB colours drawn in the original numbering, without the reorder
    to the current one, put S5 and D1-S4 in the wrong places: well past the
    thresholds the port meets."""
    _iou, d = _against_matlab("cerebellum", reorder=False)
    assert (d > 0.1).mean() > 0.05


def test_macaque_flatmap_lines_up():
    """The macaque flatmap (tools/build_macaque_flatmap.py): every level's labels cover the map, left on the left,
    the right hemisphere the left's mirror image, and each SARM6 lobule inside its SARM5 / SARM4 / SARM3 parent."""
    six = _atlas.macaque_flatmap("SARM6")
    pts, faces, grey, lab6 = six
    assert faces.min() == 0 and faces.max() == len(pts) - 1 and 0 <= grey.min() and grey.max() <= 1
    names6 = _atlas.macaque_cerebellum_names("SARM6")[0]
    assert sorted(np.unique(lab6).tolist()) == list(range(1, len(names6) + 1))
    left = np.array([names6[k - 1].startswith("SL_") for k in lab6])
    assert (pts[left, 0] < 0).all() and (pts[~left, 0] > 0).all()
    # mirror: each left vertex has a right vertex at (-x, y) with the same lobule (the template is symmetric)
    from scipy.spatial import cKDTree
    d, j = cKDTree(pts[~left]).query(pts[left] * [-1, 1])
    assert d.max() < 1e-3
    assert [names6[k - 1][3:] for k in lab6[left]] == [names6[k - 1][3:] for k in lab6[~left][j]]
    for atlas in ("SARM5", "SARM4", "SARM3"):
        _p, _f, _g, lab = _atlas.macaque_flatmap(atlas)
        names = _atlas.macaque_cerebellum_names(atlas)[0]
        for k in np.unique(lab6):                               # one parent per lobule, on its side
            parents = {names[i - 1] for i in np.unique(lab[lab6 == k])}
            assert len(parents) == 1 and next(iter(parents))[:3] == names6[k - 1][:3]


def test_macaque_map_renders_symmetric_values_symmetrically():
    names = scmaps.cerebellum_regions("SARM6")
    v = {n: (k % 21) - 10.0 for k, n in enumerate(names)}       # the same value on both sides
    img = scmaps.cerebellum_map(v, atlas="SARM6", cmap="RdBu_r", vmin=-10, vmax=10, dpi=60)
    assert img.ndim == 3 and img.shape[2] == 4 and (img[..., 3] > 0).mean() > 0.3
    a = img[..., 3] > 0
    cols = np.nonzero(a.any(0))[0]
    mid = (cols.min() + cols.max()) / 2
    assert abs(np.nonzero(a)[1].mean() - mid) < 2                  # the drawing is centred
    flip = img[:, ::-1]
    common = (img[..., 3] == 255) & (flip[..., 3] == 255)
    assert np.abs(img[common, :3].astype(int) - flip[common, :3].astype(int)).mean() < 6
    with pytest.raises(ValueError):
        C.render_cerebellum(np.zeros((32, 3)), atlas="SARM6")


# --- the solid style ----------------------------------------------------------

def test_solid_style_keeps_every_region():
    """The smoothing of the borders loses no region on any flatmap (SUIT's
    flatmap has a 2-vertex region, D1R) and changes the outline little: the
    solid silhouette is the suit style's."""
    from scmaps import _solid
    for atlas in scmaps.CEREBELLUM_ATLASES:
        _sil, _edge, regions, borders = _solid.geometry(atlas)
        assert len(regions) == C.n_regions(atlas) and all(regions) and borders, atlas
    for atlas, n in (("NettekovenAsym32", 32), ("SARM5", 10)):
        rgb = np.tile([0.3, 0.5, 0.7], (n, 1))
        _M, _P, iou = _aligned(C.render_cerebellum(rgb, 0.75, dpi=60, atlas=atlas),
                               C.render_cerebellum(rgb, dpi=60, atlas=atlas, style="solid"))
        assert iou > 0.97, atlas


def test_solid_style_draws_flat_colours_the_lines_and_the_outline():
    from scipy.ndimage import binary_erosion
    names = scmaps.cerebellum_regions("SARM5")
    rgb = np.full((len(names), 3), np.nan)
    rgb[[names.index("SL_PVCbCx"), names.index("SR_PVCbCx")]] = [0.8, 0.1, 0.1]
    rgb[names.index("SL_LCbCx")] = [0.1, 0.2, 0.7]
    from scmaps import _solid
    canvas = dict(dpi=100, atlas="SARM5", fig_in=C.FLAT_FIG_IN, lim=C.FLAT_LIM)      # uncropped: one pixel grid
    img = _solid.render_solid(rgb, **canvas)
    plain = _solid.render_solid(rgb, **canvas, border=None, outline=None)

    def share(im, c):
        px = im[im[..., 3] == 255][:, :3].astype(int)
        return (np.abs(px - np.round(np.asarray(c) * 255)).max(1) <= 1).mean()
    for im in (img, plain):                         # unshaded colours; light grey without a value
        assert share(im, [0.8, 0.1, 0.1]) > 0.05 and share(im, [0.1, 0.2, 0.7]) > 0.05
        assert share(im, [0.85] * 3) > 0.4
    assert share(img, [1.0] * 3) > 0.002 and share(plain, [1.0] * 3) == 0      # the white lines
    a = plain[..., 3] > 0
    rim = a & ~binary_erosion(a, iterations=2)
    grey = rim & (np.abs(plain[..., :3].astype(int) - 217).max(-1) <= 2)    # the rim of the regions without a value
    assert grey.sum() > 50 and img[grey, :3].mean() < plain[grey, :3].mean() - 10   # the grey outline over it


def test_solid_style_renders_symmetric_values_symmetrically():
    names = scmaps.cerebellum_regions("SARM6")
    v = {n: (k % 21) - 10.0 for k, n in enumerate(names)}
    img = scmaps.cerebellum_map(v, atlas="SARM6", cmap="RdBu_r", vmin=-10, vmax=10, dpi=60, style="solid")
    flip = img[:, ::-1]
    common = (img[..., 3] == 255) & (flip[..., 3] == 255)
    assert common.mean() > 0.3
    assert np.abs(img[common, :3].astype(int) - flip[common, :3].astype(int)).mean() < 6
    with pytest.raises(ValueError):
        C.render_cerebellum(np.zeros((42, 3)), atlas="SARM6", style="flat")
