"""The macaque subcortex (NMT v2.1-sym, SARM): random values on SARM level 3's
31 regions per hemisphere and level 5's 156, each in the row layout (left
lateral, left medial, right medial, right lateral) with its views cut away
below (cut=True): the lateral views without the outer structures, the medial
views without the midline ones.

    python examples/macaque_subcortex.py      # writes examples/macaque_subcortex.png
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import scmaps

matplotlib.use("Agg")
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 1, figsize=(8, 6.6), layout="constrained")
style = dict(cmap="RdBu_r", vmin=-2, vmax=2, dpi=150, layout="row", cut=True)
for ax, atlas in zip(axes, ("SARM3", "SARM5")):
    names = scmaps.subcortex_regions(atlas)
    sm = scmaps.plot_subcortex(dict(zip(names, rng.normal(size=len(names)))), ax=ax, atlas=atlas, **style)
    ax.set_title(f'atlas="{atlas}": {len(names) // 2} regions per hemisphere; below, cut away', fontsize=9)
fig.colorbar(sm, ax=axes, shrink=0.5, label="value")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "macaque_subcortex.png"), dpi=150)
