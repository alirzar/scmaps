"""Cerebellar regions on a flatmap, drawn as SUIT's plotflatmap.m draws them:
the 32 Nettekoven Asym32 regions on SUIT's flatmap (human), or SARM's
cerebellar cortex regions at one of its levels on the macaque flatmap built
from NMT v2.1-sym (tools/build_macaque_flatmap.py).

The flatmap (FLAT.surf.gii) and its underlay (SUIT.shape.gii; the macaque's
NMT.shape.gii has SUIT's value distribution) coloured with plotflatmap.m's
arithmetic: the underlay through gray(256) indexed on [-1, 0.5], the overlay
blended as colour .* ((1 - alpha) + alpha .* underlay) with one opacity per
region (0 = the colour unshaded, e.g. a seed), a region without a value
showing the underlay alone, and the colours interpolated across each
triangle (MATLAB's 'interp'). The canvas is FLAT_FIG_IN inches square
spanning [-FLAT_LIM, FLAT_LIM] on both axes -- the figure plotCerebellum.m
draws -- so `dpi` / 50 pixels per flatmap unit; both flatmaps span about
+-90 units.

That is style "suit". Style "solid" (_solid.py) draws on the same canvas the
regions in flat colour on a light-grey background (a region without a
value), white lines between the regions and a grey outline.
"""

import numpy as np

from ._atlas import MACAQUE_ATLASES, flatmap, macaque_cerebellum_names, macaque_flatmap
from ._solid import BORDER, NAN_RGB, OUTLINE, render_solid
from ._vtk import crop, draw_unlit

ALPHA = 0.75                # overlay opacity on the underlay, as in the papers
FLAT_FIG_IN = 4.0           # the figure, inches square, the axes filling it
FLAT_LIM = 100.0            # xlim = ylim = [-100, 100]
HUMAN = "NettekovenAsym32"
ATLASES = (HUMAN, *MACAQUE_ATLASES)
STYLES = ("suit", "solid")


def _flatmap(atlas):
    if atlas == HUMAN:
        return flatmap()
    if atlas in MACAQUE_ATLASES:
        return macaque_flatmap(atlas)
    raise ValueError(f"unknown cerebellar atlas {atlas!r}; one of {ATLASES}")


def n_regions(atlas=HUMAN):
    """The number of regions of a cerebellar atlas (32 for Nettekoven Asym32)."""
    _flatmap(atlas)
    return 32 if atlas == HUMAN else len(macaque_cerebellum_names(atlas)[0])


def flat_colours(rgb, alphas, atlas=HUMAN):
    """Per-vertex colour, plotflatmap's blend: overlay .* ((1 - a) + a .*
    underlay); a vertex whose region has no value shows the underlay alone."""
    _pts, _faces, grey, lab = _flatmap(atlas)
    rgb = np.asarray(rgb, float).reshape(-1, 3)
    if len(rgb) != n_regions(atlas):
        raise ValueError(f"{atlas}: expected {n_regions(atlas)} region colours, got {len(rgb)}")
    o = rgb[lab - 1]
    a = np.asarray(alphas, float).reshape(len(rgb))[lab - 1][:, None]
    g = grey[:, None]
    return np.where(np.isnan(o), np.repeat(g, 3, axis=1), o * ((1.0 - a) + a * g))


def render_cerebellum(rgb, alphas=ALPHA, dpi=300, atlas=HUMAN, style="suit", nan_rgb=NAN_RGB, border=BORDER,
                      outline=OUTLINE):
    """RGBA image (uint8, transparent background, cropped to the drawing) of
    the atlas's flatmap with its n regions coloured `rgb` (n x 3 by region,
    NaN rows = no value). Style "suit": `alphas` the overlay opacity, one
    value or one per region. Style "solid": no value drawn `nan_rgb`, the
    lines between the regions `border` and the outline `outline` (a colour,
    or None for none)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; one of {STYLES}")
    rgb = np.asarray(rgb, float).reshape(-1, 3)
    if len(rgb) != n_regions(atlas):
        raise ValueError(f"{atlas}: expected {n_regions(atlas)} region colours, got {len(rgb)}")
    if style == "solid":
        return crop(render_solid(rgb, dpi, atlas, FLAT_FIG_IN, FLAT_LIM, nan_rgb, border, outline))
    a = np.broadcast_to(np.asarray(alphas, float), (len(rgb),))
    pts, faces, _grey, _lab = _flatmap(atlas)
    n = int(round(FLAT_FIG_IN * dpi))
    img = draw_unlit(np.c_[pts, np.zeros(len(pts))], faces, flat_colours(rgb, a, atlas), n, n,
                     focal=(0.0, 0.0, 0.0), position=(0.0, 0.0, 500.0), view_up=(0, 1, 0),
                     half_height=FLAT_LIM)
    return crop(img)
