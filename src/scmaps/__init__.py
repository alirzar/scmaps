"""scmaps: subcortical (Tian S2) and cerebellar (Nettekoven Asym32, SUIT
flatmap) maps in Python, drawn as the MATLAB plotSubcortex.m / SUIT
plotflatmap.m draw them.

    import scmaps
    img = scmaps.subcortex_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    img = scmaps.cerebellum_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    scmaps.show(img, ax)

See `scmaps.SUBCORTEX_REGIONS` and `scmaps.CEREBELLUM_REGIONS` for the order of
the 32 values of each, and `scmaps.CEREBELLUM_REGIONS_ORIGINAL` for cerebellar
data in the Nettekoven atlas's original (pre-November 2023) numbering.
"""

from .cerebellum import render_cerebellum
from .colours import CEREBELLUM_REGIONS, CEREBELLUM_REGIONS_ORIGINAL, SUBCORTEX_REGIONS
from .maps import cerebellum_map, plot_cerebellum, plot_subcortex, show, subcortex_map
from .subcortex import LAYOUTS as SUBCORTEX_LAYOUTS
from .subcortex import render_subcortex

__version__ = "0.1.0"

__all__ = ["subcortex_map", "cerebellum_map", "plot_subcortex", "plot_cerebellum", "show",
           "render_subcortex", "render_cerebellum", "SUBCORTEX_REGIONS", "CEREBELLUM_REGIONS",
           "CEREBELLUM_REGIONS_ORIGINAL", "SUBCORTEX_LAYOUTS"]
