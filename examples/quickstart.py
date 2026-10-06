"""Draw random values on the subcortex and the cerebellum, side by side with a
shared colourbar, and highlight two cerebellar regions as a seed.

    python examples/quickstart.py      # writes examples/quickstart.png
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import scmaps

matplotlib.use("Agg")
rng = np.random.default_rng(0)
sub = rng.normal(size=32)
cer = rng.normal(size=32)
sub[scmaps.SUBCORTEX_REGIONS.index("aHIP-lh")] = np.nan       # no value: grey

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), width_ratios=(1.4, 1.2, 1.0),
                         layout="constrained")
style = dict(cmap="RdBu_r", vmin=-2, vmax=2, dpi=200)
sm = scmaps.plot_subcortex(sub, ax=axes[0], layout="grid", **style)
scmaps.plot_cerebellum(cer, ax=axes[1], **style)
scmaps.plot_cerebellum(colors="0.6", ax=axes[2], highlight=["M3L", "M3R"], dpi=200)
axes[0].set_title("subcortex (grid)", fontsize=9)
axes[1].set_title("cerebellum", fontsize=9)
axes[2].set_title("a seed, highlighted", fontsize=9)
fig.colorbar(sm, ax=axes[:2], shrink=0.6, label="value")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quickstart.png"), dpi=150)
