"""The 'solid' style of the cerebellar flatmaps: the regions in flat colour on a
solid light-grey background, white lines between the regions and a grey
outline, drawn as vector shapes by matplotlib (Agg) on the canvas the 'suit'
style draws on (FLAT_FIG_IN inches square spanning +-FLAT_LIM flat units). It
needs no OpenGL.

The shapes, once per atlas: each region's indicator (1 at its vertices,
interpolated across the triangles) is sampled on a GRID-unit grid and
smoothed (Gaussian, SMOOTH flat units), and every grid point goes to the
region with the largest. A region is the filled contour of its lead over the
next region; the borders are those leads' zero lines; the outline is the
half level of the smoothed outside. So the borders and the outline run
smooth instead of along the triangles: the macaque flatmap is a grid (0.95
units) whose label borders step, smoothed over 0.8 of its cells (no lobule
loses more than 5 % of its area); SUIT's flatmap is irregular and carries
tiny regions (D1R: 2 vertices), so only a touch, which keeps every region.
"""

from functools import lru_cache

import numpy as np

GRID = 0.25                          # the regions' fields: grid spacing, flat units
SMOOTH = {"NettekovenAsym32": 0.2}   # Gaussian sigma of the borders' smoothing, flat units
MACAQUE_SMOOTH = 0.76                # ... on the macaque flatmap: 0.8 of its grid cells
NAN_RGB = (0.85, 0.85, 0.85)         # a region without a value: light grey
BORDER = "white"                     # the lines between the regions
OUTLINE = "#a6a6a6"                  # the outline, so near-white colours keep an edge
BORDER_PT, OUTLINE_PT = 0.9, 0.6     # their widths, points on the canvas
SEAM_PT = 0.25                       # each fill's edge in its own colour: no seam between neighbours


def _gaussian(a, sigma, cval=0.0):
    """scipy.ndimage.gaussian_filter(a, sigma, mode="constant", cval=cval)
    (its kernel, truncated at 4 sigma) in numpy: scipy is not a dependency."""
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(np.asarray(a, float), r, constant_values=cval)
    h, w = a.shape
    t = sum(k[i] * p[i:i + h] for i in range(2 * r + 1))
    return sum(k[j] * t[:, j:j + w] for j in range(2 * r + 1))


def _rasterise(pts, faces, x0, y0, h, ny, nx):
    """For every point (x0 + i h, y0 + j h) of an ny x nx grid: the triangle
    that holds it (-1: none; where triangles overlap, as at a fold of SUIT's
    flatmap, the last) and its barycentric weights."""
    P = pts[faces]
    lo = np.ceil((P.min(1) - (x0, y0)) / h - 1e-9).astype(int)
    size = np.floor((P.max(1) - (x0, y0)) / h + 1e-9).astype(int) - lo + 1
    tri = np.full((ny, nx), -1)
    w = np.zeros((ny, nx, 3))
    a = P[:, 0]
    e1, e2 = P[:, 1] - a, P[:, 2] - a
    den = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    for di in range(size[:, 0].max()):
        for dj in range(size[:, 1].max()):
            t = np.nonzero((size[:, 0] > di) & (size[:, 1] > dj))[0]
            i, j = lo[t, 0] + di, lo[t, 1] + dj
            px, py = x0 + i * h - a[t, 0], y0 + j * h - a[t, 1]
            l1 = (px * e2[t, 1] - e2[t, 0] * py) / den[t]
            l2 = (e1[t, 0] * py - px * e1[t, 1]) / den[t]
            l0 = 1.0 - l1 - l2
            ok = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            tri[j[ok], i[ok]] = t[ok]
            w[j[ok], i[ok]] = np.c_[l0[ok], l1[ok], l2[ok]]
    return tri, w


@lru_cache(maxsize=None)
def geometry(atlas):
    """(silhouette, outline, regions, borders) of an atlas's flatmap, flat
    units: the silhouette a compound Path, the outline and the borders lists
    of polylines (each border once), regions one list of Paths per region
    (1..n in order)."""
    import contourpy
    from matplotlib.path import Path

    from .cerebellum import _flatmap, n_regions
    pts, faces, _grey, lab = _flatmap(atlas)
    n = n_regions(atlas)
    s = SMOOTH.get(atlas, MACAQUE_SMOOTH)
    m = 4 * s + 2 * GRID
    x = np.arange(pts[:, 0].min() - m, pts[:, 0].max() + m + GRID, GRID)
    y = np.arange(pts[:, 1].min() - m, pts[:, 1].max() + m + GRID, GRID)
    tri, w = _rasterise(pts, faces, x[0], y[0], GRID, len(y), len(x))
    inside = tri >= 0
    vl = np.zeros(tri.shape + (3,), int)            # the labels of each grid point's triangle's vertices
    vl[inside] = lab[faces[tri[inside]]]

    reach = int(4.0 * s / GRID + 0.5)              # the Gaussian's, grid points

    def field(k):
        """Region k's smoothed indicator where it is not 0: (that part of the grid, the field there)."""
        on = vl == k
        hit = on.any(2)
        rows, cols = np.nonzero(hit.any(1))[0], np.nonzero(hit.any(0))[0]
        part = (slice(max(rows[0] - reach, 0), rows[-1] + reach + 1),
                slice(max(cols[0] - reach, 0), cols[-1] + reach + 1))
        return part, _gaussian((w[part] * on[part]).sum(2), s / GRID)

    best, second = np.zeros(tri.shape), np.zeros(tri.shape)        # the largest field and the next, and
    arg, arg2 = np.zeros(tri.shape, int), np.zeros(tri.shape, int)  # their regions (0: none)
    for k in range(1, n + 1):
        part, f = field(k)
        b, sd, a, a2 = best[part], second[part], arg[part], arg2[part]
        up, up2 = f > b, (f <= b) & (f > sd)
        second[part], arg2[part] = np.where(up, b, np.where(up2, f, sd)), np.where(up, a, np.where(up2, k, a2))
        best[part], arg[part] = np.where(up, f, b), np.where(up, k, a)
    half = 0.5 - _gaussian((~inside).astype(float), s / GRID, cval=1.0)
    gen = contourpy.contour_generator(x, y, half, fill_type="OuterCode", line_type="Separate")
    silhouette = Path.make_compound_path(*(Path(p, c) for p, c in zip(*gen.filled(0.0, 1.0))))
    outline = gen.lines(0.0)
    regions, borders = [], []
    for k in range(1, n + 1):
        part, f = field(k)
        lead = np.where(arg[part] == k, best[part] - second[part], f - best[part])
        lead[best[part] == 0] = -1.0                 # no region at all
        gen = contourpy.contour_generator(x[part[1]], y[part[0]], lead, fill_type="OuterCode", line_type="Separate")
        regions.append([Path(p, c) for p, c in zip(*gen.filled(1e-9, 2.0))])
        for line in gen.lines(0.0):                  # kept where the region across has the larger label
            c = np.clip(np.round((line[:, 0] - x[0]) / GRID).astype(int), 0, len(x) - 1)
            r = np.clip(np.round((line[:, 1] - y[0]) / GRID).astype(int), 0, len(y) - 1)
            ok = np.where(arg[r, c] == k, arg2[r, c], arg[r, c]) > k
            d = np.diff(np.r_[0, ok.astype(int), 0])
            borders += [line[a:b] for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= 2]
    return silhouette, outline, regions, borders


def render_solid(rgb, dpi, atlas, fig_in, lim, nan_rgb=NAN_RGB, border=BORDER, outline=OUTLINE):
    """RGBA image (uint8, transparent background, uncropped) of the atlas's
    flatmap in the solid style: its n regions coloured `rgb` (n x 3, NaN rows
    = no value, drawn `nan_rgb`); `border` / `outline` a colour or None (no
    lines)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    silhouette, edge, regions, borders = geometry(atlas)
    fig = Figure(figsize=(fig_in, fig_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_axis_off()
    clip = PathPatch(silhouette, facecolor=nan_rgb, edgecolor="none")
    ax.add_patch(clip)
    for paths, c in zip(regions, rgb):
        if paths and not np.isnan(c).any():
            ax.add_collection(PathCollection(paths, facecolors=[c], edgecolors=[c], linewidths=SEAM_PT,
                                             clip_path=clip))
    if border is not None:
        ax.add_collection(LineCollection(borders, colors=[border], linewidths=BORDER_PT, capstyle="round",
                                         joinstyle="round", clip_path=clip))
    if outline is not None:
        ax.add_collection(LineCollection(edge, colors=[outline], linewidths=OUTLINE_PT, joinstyle="round"))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()
