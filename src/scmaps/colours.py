"""Region names, and values or colours -> the 32 x 3 RGB the renderers take.

Regions are in plotting order:

* subcortex: the 16 Tian S2 (3T) ROIs of the left hemisphere, then the same
  16 of the right (`SUBCORTEX_REGIONS`, e.g. "aHIP-lh" ... "pCAU-rh");
* cerebellum: the 32 regions of the Nettekoven asymmetric atlas
  (NettekovenAsym32) by their label in the atlas's current numbering, 1-16
  left, 17-32 right (`CEREBELLUM_REGIONS`, "M1L" ... "S5L", "M1R" ... "S5R").

Every function taking `values` accepts a length-32 sequence in that order, or
a mapping (dict, pandas Series) from region name to value; a region missing
from a mapping, or NaN, has no value.

Data extracted with the atlas's original numbering (atlas files from before
November 2023), where today's S5 was A4 with label 8 (24 on the right) and
D1-S4 were 9-16 (25-32), are in `CEREBELLUM_REGIONS_ORIGINAL` order: pass them
by name, `dict(zip(CEREBELLUM_REGIONS_ORIGINAL, values))`.
"""

from collections.abc import Mapping

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgb

from ._atlas import cerebellum_names, subcortex_names

SUBCORTEX_REGIONS = subcortex_names()
CEREBELLUM_REGIONS = cerebellum_names()
# the regions of labels 1-32 in the atlas's original release, by today's names
CEREBELLUM_REGIONS_ORIGINAL = tuple(f"{r}{h}" for h in "LR" for r in (
    "M1", "M2", "M3", "M4", "A1", "A2", "A3", "S5", "D1", "D2", "D3", "D4", "S1", "S2", "S3", "S4"))
HIGHLIGHT = "#FFD700"


def _is_mapping(x):
    """A dict-like or a pandas Series (labelled by region name)."""
    return isinstance(x, Mapping) or (hasattr(x, "index") and hasattr(x, "items"))


def _by_name(mapping, names, convert, empty):
    unknown = set(mapping) - set(names)
    if unknown:
        raise KeyError(f"unknown region name(s) {sorted(map(str, unknown))}; "
                       f"expected names like {names[0]!r}")
    return [convert(mapping[n]) if n in mapping else empty for n in names]


def as_values(values, names):
    """Length-32 float vector in plotting order (NaN = no value)."""
    if values is None:
        return np.full(32, np.nan)
    if _is_mapping(values):
        values = _by_name(dict(values.items()), names, float, np.nan)
    v = np.asarray(values, float).ravel()
    if v.shape != (32,):
        raise ValueError(f"expected 32 values, got {v.size}")
    return v


def as_mask(regions, names):
    """32 bools from a boolean sequence of 32, or an iterable of region names."""
    if regions is None:
        return np.zeros(32, bool)
    regions = list(regions)
    if len(regions) == 32 and all(isinstance(r, (bool, np.bool_)) for r in regions):
        return np.asarray(regions, bool)
    unknown = set(regions) - set(names)
    if unknown:
        raise KeyError(f"unknown region name(s) {sorted(map(str, unknown))}")
    return np.isin(np.asarray(names), regions)


def _rgb_or_nan(c):
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return (np.nan, np.nan, np.nan)
    return to_rgb(c)


def as_colours(colours, names):
    """32 x 3 RGB from one matplotlib colour, 32 colours (None = no value) or
    a mapping from region name to colour."""
    if _is_mapping(colours):
        return np.asarray(_by_name(dict(colours.items()), names, _rgb_or_nan, (np.nan,) * 3),
                          float)
    if isinstance(colours, str) or (is_color_like(colours) and np.ndim(colours) == 1):
        return np.tile(to_rgb(colours), (32, 1))
    rgb = np.asarray([_rgb_or_nan(c) for c in colours], float)
    if rgb.shape != (32, 3):
        raise ValueError(f"expected one colour or 32, got {len(rgb)}")
    return rgb


def value_range(values, vmin=None, vmax=None):
    """(vmin, vmax), each defaulting to the finite values' min / max."""
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    return vmin, vmax


def values_to_rgb(values, cmap="viridis", vmin=None, vmax=None):
    """32 x 3 RGB through `cmap` on [vmin, vmax]: a value beyond the range takes
    the end colour, NaN stays NaN (no value)."""
    v = np.asarray(values, float)
    cm = cmap if isinstance(cmap, Colormap) else colormaps[cmap]
    norm = Normalize(*value_range(v, vmin, vmax), clip=True)
    rgb = np.asarray(cm(norm(np.nan_to_num(v))))[:, :3].copy()
    rgb[np.isnan(v)] = np.nan
    return rgb
