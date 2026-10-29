"""scmaps: subcortical (Tian S2) and cerebellar (Nettekoven Asym32, SUIT
flatmap) maps in Python, drawn as the MATLAB plotSubcortex.m / SUIT
plotflatmap.m draw them; and macaque cerebellar maps (SARM, levels 3-6) on a
flatmap built the same way from the NMT v2.1-sym template.

    import scmaps
    img = scmaps.subcortex_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    img = scmaps.cerebellum_map(values32, cmap="RdBu_r", vmin=-1, vmax=1)
    img = scmaps.cerebellum_map({"SL_Cb6V": 1.0}, atlas="SARM6")   # macaque
    img = scmaps.cerebellum_map(values32, style="solid")   # flat colours, light grey, white borders
    scmaps.show(img, ax)

See `scmaps.SUBCORTEX_REGIONS` and `scmaps.CEREBELLUM_REGIONS` for the order of
the 32 values of each, `scmaps.CEREBELLUM_REGIONS_ORIGINAL` for cerebellar
data in the Nettekoven atlas's original (pre-November 2023) numbering, and
`scmaps.cerebellum_regions(atlas)` for the macaque atlases' regions
(`scmaps.CEREBELLUM_ATLASES`).
"""

from .cerebellum import ATLASES as CEREBELLUM_ATLASES
from .cerebellum import STYLES as CEREBELLUM_STYLES
from .cerebellum import render_cerebellum
from .colours import (CEREBELLUM_REGIONS, CEREBELLUM_REGIONS_ORIGINAL, MACAQUE_CEREBELLUM_REGIONS,
                      SUBCORTEX_REGIONS, cerebellum_regions)
from .maps import cerebellum_map, plot_cerebellum, plot_subcortex, show, subcortex_map
from .subcortex import LAYOUTS as SUBCORTEX_LAYOUTS
from .subcortex import render_subcortex

__version__ = "0.2.0"

__all__ = ["subcortex_map", "cerebellum_map", "plot_subcortex", "plot_cerebellum", "show",
           "render_subcortex", "render_cerebellum", "SUBCORTEX_REGIONS", "CEREBELLUM_REGIONS",
           "CEREBELLUM_REGIONS_ORIGINAL", "SUBCORTEX_LAYOUTS", "CEREBELLUM_ATLASES",
           "MACAQUE_CEREBELLUM_REGIONS", "cerebellum_regions", "CEREBELLUM_STYLES"]
