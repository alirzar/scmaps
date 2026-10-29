"""The public API: values (or colours) in, an RGBA image or a matplotlib axes
out.

    import scmaps
    img = scmaps.subcortex_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    img = scmaps.cerebellum_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    img = scmaps.cerebellum_map({"SL_Cb6V": 1.0}, atlas="SARM6")   # macaque
    sm = scmaps.plot_subcortex(values32, ax=ax, cmap="RdBu_r")  # for a colourbar
"""

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb

from .cerebellum import ALPHA, BORDER, HUMAN, NAN_RGB, OUTLINE, render_cerebellum
from .colours import (HIGHLIGHT, SUBCORTEX_REGIONS, as_colours, as_mask, as_values, cerebellum_regions,
                      value_range, values_to_rgb)
from .subcortex import GREY, render_subcortex

DPI = 300


def _rgb(values, colors, names, cmap, vmin, vmax, highlight, highlight_color):
    """(n x 3 RGB, highlight mask)."""
    if values is not None and colors is not None:
        raise ValueError("give `values` or `colors`, not both")
    if colors is not None:
        rgb = as_colours(colors, names)
    else:
        rgb = values_to_rgb(as_values(values, names), cmap, vmin, vmax)
    mask = as_mask(highlight, names)
    rgb[mask] = to_rgb(highlight_color)
    return rgb, mask


def subcortex_map(values=None, *, colors=None, cmap="viridis", vmin=None, vmax=None,
                  layout="grid", dpi=DPI, highlight=None, highlight_color=HIGHLIGHT,
                  nan_color=(GREY, GREY, GREY)):
    """RGBA image (H x W x 4 uint8, transparent background) of the 32 Tian S2
    subcortical ROIs.

    values : 32 values in `SUBCORTEX_REGIONS` order, or a mapping from region
        name to value. NaN / missing = no value, drawn `nan_color`.
    colors : instead of values, one colour, 32 colours, or a mapping from
        region name to colour.
    cmap, vmin, vmax : the colour map and its range (default: the values' min
        and max); values beyond the range take the end colour.
    layout : "grid" (2 x 2), "row" (1 x 4), "lateral" (both lateral faces) or
        "left" (the left hemisphere's lateral face).
    dpi : each view tile is 1.2 inches tall at this resolution.
    highlight : region names (or 32 bools) painted `highlight_color`.
    """
    rgb, _mask = _rgb(values, colors, SUBCORTEX_REGIONS, cmap, vmin, vmax,
                      highlight, highlight_color)
    return render_subcortex(rgb, layout, dpi, nan_rgb=to_rgb(nan_color))


def cerebellum_map(values=None, *, colors=None, cmap="viridis", vmin=None, vmax=None,
                   alpha=ALPHA, dpi=DPI, highlight=None, highlight_color=HIGHLIGHT,
                   atlas=HUMAN, style="suit", nan_color=NAN_RGB, border_color=BORDER,
                   outline_color=OUTLINE):
    """RGBA image (H x W x 4 uint8, transparent background) of cerebellar
    regions on a flatmap: the 32 Nettekoven Asym32 regions on SUIT's flatmap
    (atlas="NettekovenAsym32", human, the default), or SARM's cerebellar cortex
    regions on the macaque flatmap (atlas="SARM3", "SARM4", "SARM5" or
    "SARM6": 1, 4, 5 or 21 regions per hemisphere).

    values, colors, cmap, vmin, vmax : as for `subcortex_map`, by
        `cerebellum_regions(atlas)` (human: `CEREBELLUM_REGIONS`, the atlas's
        current label order). A region without a value shows the flatmap's
        grey underlay alone (style "solid": `nan_color`). Nettekoven values
        in the atlas's original (pre-November 2023) order: pass
        `dict(zip(CEREBELLUM_REGIONS_ORIGINAL, values))`.
    alpha : how much of the underlay's shading shows through the colour (0 =
        flat colour, 1 = fully shaded), one value or one per region.
    dpi : the flatmap canvas is 4 inches square at this resolution.
    highlight : region names (or one bool per region) painted
        `highlight_color` and drawn unshaded, e.g. a seed.
    atlas : the parcellation, and with it the flatmap (human or macaque).
    style : "suit" (SUIT's plotflatmap look: the colours blended with the
        grey underlay's shading) or "solid" (the colours flat on a solid
        background, white lines between the regions, a grey outline; the
        borders smoothed).
    nan_color, border_color, outline_color : style "solid": the colour of a
        region without a value (light grey), of the lines between the regions
        (white) and of the outline (grey); None draws no lines / no outline.
    """
    names = cerebellum_regions(atlas)
    rgb, mask = _rgb(values, colors, names, cmap, vmin, vmax, highlight, highlight_color)
    alphas = np.broadcast_to(np.asarray(alpha, float), (len(names),)).copy()
    alphas[mask] = 0.0
    return render_cerebellum(rgb, alphas, dpi, atlas=atlas, style=style, nan_rgb=to_rgb(nan_color),
                             border=None if border_color is None else to_rgb(border_color),
                             outline=None if outline_color is None else to_rgb(outline_color))


def show(img, ax=None):
    """Draw a rendered image in `ax` (default: the current axes), axes off.
    Returns the AxesImage."""
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    im = ax.imshow(img, interpolation="antialiased")
    ax.set_axis_off()
    return im


def plot_subcortex(values=None, ax=None, *, colors=None, cmap="viridis", vmin=None, vmax=None,
                   **kwargs):
    """`subcortex_map` drawn into `ax`. Returns a ScalarMappable for a colourbar
    (None when `colors` are given)."""
    show(subcortex_map(values, colors=colors, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs), ax)
    return _mappable(values, colors, SUBCORTEX_REGIONS, cmap, vmin, vmax)


def plot_cerebellum(values=None, ax=None, *, colors=None, cmap="viridis", vmin=None, vmax=None,
                    atlas=HUMAN, **kwargs):
    """`cerebellum_map` drawn into `ax`. Returns a ScalarMappable for a
    colourbar (None when `colors` are given)."""
    show(cerebellum_map(values, colors=colors, cmap=cmap, vmin=vmin, vmax=vmax, atlas=atlas, **kwargs), ax)
    return _mappable(values, colors, cerebellum_regions(atlas), cmap, vmin, vmax)


def _mappable(values, colors, names, cmap, vmin, vmax):
    if colors is not None:
        return None
    v = as_values(values, names)
    return ScalarMappable(norm=Normalize(*value_range(v, vmin, vmax)), cmap=cmap)
