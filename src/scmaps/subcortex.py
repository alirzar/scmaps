"""The 32 Tian S2 subcortical ROIs drawn as plotSubcortex.m draws them.

The ROI surfaces (MATLAB's `boundary(..., 0.5)` of each ROI's voxels) in
plotSubcortex.m's views and geometry: per layout the same hemisphere / side
views (`VIEWS`), tiles edge to edge, one scale for every view -- each
hemisphere centred in its own box, every box with the larger half-extent of
the two hemispheres plus 1.5 voxels -- orthographic side views with the box's
z-extent filling a TILE_IN-inch tile.

The lighting is MATLAB's, computed per vertex (`lit_colours`) and drawn unlit
with the colours interpolated across each triangle, which is what MATLAB's
`lighting gouraud` does. Its constants were measured on figures MATLAB R2024b
made: vertex normals are the sum of the adjacent faces' UNIT normals,
BackFaceLighting 'reverselit', the camera at CAM_DIST x the box's
half-diagonal, `camlight` a LOCAL light LIGHT_AZ right and LIGHT_EL up of the
camera at the camera's distance, and `material dull` = AMBIENT + DIFFUSE x cos,
no specular. An ROI without a value is grey (GREY).
"""

import numpy as np

from ._atlas import subcortex_frame, subcortex_meshes
from ._vtk import crop, draw_unlit

GREY = 0.82                 # an ROI without a value
TILE_IN = 1.2               # tile height, inches
CAM_DIST = 10.0             # camera distance / the box's half-diagonal
LIGHT_AZ = LIGHT_EL = 30.0  # camlight: degrees right / up of the camera
AMBIENT, DIFFUSE = 0.3, 0.8  # material dull, no specular

# per layout: (hemisphere, camera side on x) per tile, rows, columns. From -x
# (MATLAB view(-90, 0)) the left hemisphere shows its lateral face.
VIEWS = {"row": ([("L", -1), ("L", +1), ("R", -1), ("R", +1)], 1, 4),
         "grid": ([("L", -1), ("R", +1), ("L", +1), ("R", -1)], 2, 2),
         "lateral": ([("L", -1), ("R", +1)], 1, 2),
         "left": ([("L", -1)], 1, 1)}
LAYOUTS = tuple(VIEWS)


def vertex_normals(pts, faces):
    """MATLAB's patch VertexNormals, normalised: the sum of the adjacent
    faces' UNIT normals (not area-weighted)."""
    p0, p1, p2 = (pts[faces[:, k]] for k in range(3))
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(pts, dtype=float)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def light_position(centre, side):
    """Where MATLAB's `camlight` (right) puts its local light for a camera on
    the x axis (side -1: at -x) looking at `centre`: LIGHT_AZ right and LIGHT_EL
    up of the camera, at the camera's distance."""
    _ctr, half = subcortex_frame()
    toward = np.array([float(side), 0.0, 0.0])          # target -> camera
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-toward, up)
    az, el = np.deg2rad(LIGHT_AZ), np.deg2rad(LIGHT_EL)
    ldir = np.cos(el) * (np.cos(az) * toward + np.sin(az) * right) + np.sin(el) * up
    return np.asarray(centre, float) + CAM_DIST * np.linalg.norm(half) * ldir


def lit_colours(pts, faces, rgb, centre, side):
    """MATLAB's gouraud lighting at every vertex: a normal facing away from the
    camera is reversed ('reverselit'), the light is a point (`light_position`),
    colour x (AMBIENT + DIFFUSE x cos), clipped to [0, 1]."""
    toward = np.array([float(side), 0.0, 0.0])
    vn = vertex_normals(pts, faces)
    vn = np.where((vn @ toward)[:, None] < 0, -vn, vn)
    L = light_position(centre, side) - pts
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    cos = np.clip((vn * L).sum(axis=1), 0.0, None)
    return np.clip(np.asarray(rgb, float) * (AMBIENT + DIFFUSE * cos)[:, None], 0.0, 1.0)


def _tile(hemi, side, rgb32, tile_w, tile_h, nan_rgb):
    pts, faces, roi = subcortex_meshes()[hemi]
    ctr, half = subcortex_frame()
    col = rgb32[roi - 1].copy()
    col[np.isnan(col).any(axis=1)] = nan_rgb
    col = lit_colours(pts, faces, col, ctr[hemi], side)
    c = ctr[hemi]
    position = (c[0] + side * CAM_DIST * np.linalg.norm(half), c[1], c[2])
    return draw_unlit(pts, faces, col, tile_w, tile_h, focal=c, position=position,
                      view_up=(0, 0, 1), half_height=half[2])


def render_subcortex(rgb, layout="grid", dpi=300, nan_rgb=(GREY, GREY, GREY)):
    """RGBA image (uint8, transparent background, cropped to the drawing) of
    the 32 ROIs coloured `rgb` (32 x 3 in plotting order, NaN rows = no value,
    drawn `nan_rgb`) in `layout` (row | grid | lateral | left), the tiles edge
    to edge, each TILE_IN inches tall at `dpi`."""
    if layout not in VIEWS:
        raise ValueError(f"unknown subcortex layout {layout!r}; one of {LAYOUTS}")
    rgb32 = np.asarray(rgb, float).reshape(32, 3)
    views, nr, nc = VIEWS[layout]
    _ctr, half = subcortex_frame()
    tile_h = int(round(TILE_IN * dpi))
    tile_w = int(round(tile_h * half[1] / half[2]))      # a side view shows y by z
    canvas = np.zeros((nr * tile_h, nc * tile_w, 4), np.uint8)
    for t, (hemi, side) in enumerate(views):
        r, c = divmod(t, nc)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = _tile(
            hemi, side, rgb32, tile_w, tile_h, np.asarray(nan_rgb, float))
    return crop(canvas)
