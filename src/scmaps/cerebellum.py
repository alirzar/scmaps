"""The 32 Nettekoven Asym32 cerebellar regions on SUIT's flatmap, drawn as
SUIT's plotflatmap.m draws them.

SUIT's flatmap (FLAT.surf.gii) and its underlay (SUIT.shape.gii) coloured
with plotflatmap.m's arithmetic: the underlay through gray(256) indexed on
[-1, 0.5], the overlay blended as colour .* ((1 - alpha) + alpha .* underlay)
with one opacity per region (0 = the colour unshaded, e.g. a seed), a region
without a value showing the underlay alone, and the colours interpolated
across each triangle (MATLAB's 'interp'). The canvas is FLAT_FIG_IN inches
square spanning [-FLAT_LIM, FLAT_LIM] on both axes -- the figure
plotCerebellum.m draws -- so `dpi` / 50 pixels per flatmap unit.
"""

import numpy as np

from ._atlas import flatmap
from ._vtk import crop, draw_unlit

ALPHA = 0.75                # overlay opacity on the underlay, as in the papers
FLAT_FIG_IN = 4.0           # the figure, inches square, the axes filling it
FLAT_LIM = 100.0            # xlim = ylim = [-100, 100]


def flat_colours(rgb, alphas):
    """Per-vertex colour, plotflatmap's blend: overlay .* ((1 - a) + a .*
    underlay); a vertex whose region has no value shows the underlay alone."""
    _pts, _faces, grey, lab = flatmap()
    o = np.asarray(rgb, float).reshape(32, 3)[lab - 1]
    a = np.asarray(alphas, float).reshape(32)[lab - 1][:, None]
    g = grey[:, None]
    return np.where(np.isnan(o), np.repeat(g, 3, axis=1), o * ((1.0 - a) + a * g))


def render_cerebellum(rgb, alphas=ALPHA, dpi=300):
    """RGBA image (uint8, transparent background, cropped to the drawing) of
    the flatmap with the 32 regions coloured `rgb` (32 x 3 by region, NaN rows
    = no value); `alphas` the overlay opacity, one value or one per region."""
    a = np.broadcast_to(np.asarray(alphas, float), (32,))
    pts, faces, _grey, _lab = flatmap()
    n = int(round(FLAT_FIG_IN * dpi))
    img = draw_unlit(np.c_[pts, np.zeros(len(pts))], faces, flat_colours(rgb, a), n, n,
                     focal=(0.0, 0.0, 0.0), position=(0.0, 0.0, 500.0), view_up=(0, 1, 0),
                     half_height=FLAT_LIM)
    return crop(img)
