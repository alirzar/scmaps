"""The macaque cerebellar flatmap (NMT v2.1-sym, SARM): its 21 level-6 lobules
per hemisphere, random values on them in the default style (SUIT's shading),
and random values on the five level-5 regions (anterior and posterior vermis,
intermediate, lateral, flocculus and paraflocculus) in style "solid".

    python examples/macaque.py      # writes examples/macaque.png
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import scmaps
from scmaps import _atlas
from scmaps.cerebellum import FLAT_FIG_IN, FLAT_LIM

matplotlib.use("Agg")
DPI = 200
rng = np.random.default_rng(0)
six = scmaps.cerebellum_regions("SARM6")
five = scmaps.cerebellum_regions("SARM5")
# lobule colours: one per lobule, the same on both sides
tab = [matplotlib.colormaps["tab20"](k % 20) for k in range(20)] + [(0.35, 0.35, 0.35, 1.0)]
lobule_colours = {n: tab[k % 21] for k, n in enumerate(six)}


def label_positions(atlas, img_shape):
    """Each left region's median vertex in the cropped image's pixels (the canvas spans +-FLAT_LIM flat units over
    FLAT_FIG_IN x DPI pixels; the crop starts 4 pixels before the drawing)."""
    pts, _faces, _grey, lab = _atlas.macaque_flatmap(atlas)
    names = scmaps.cerebellum_regions(atlas)
    px = FLAT_FIG_IN * DPI / (2 * FLAT_LIM)
    x0, y1 = pts[:, 0].min(), pts[:, 1].max()
    out = {}
    for k, n in enumerate(names[:len(names) // 2]):
        p = np.median(pts[lab == k + 1], axis=0)
        out[n] = ((p[0] - x0) * px + 4, (y1 - p[1]) * px + 4)
    return out


fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
img = scmaps.cerebellum_map(colors=lobule_colours, atlas="SARM6", dpi=DPI, style="solid")
scmaps.show(img, axes[0])
for n, (c, r) in label_positions("SARM6", img.shape).items():
    axes[0].text(c, r, n[3:].replace("Cb", "").replace("V", " V").replace("I", " I"), ha="center", va="center",
                 fontsize=5, bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.2))
style = dict(cmap="RdBu_r", vmin=-2, vmax=2, dpi=DPI)
sm = scmaps.plot_cerebellum(dict(zip(six, rng.normal(size=len(six)))), ax=axes[1], atlas="SARM6", **style)
scmaps.plot_cerebellum(dict(zip(five, rng.normal(size=len(five)))), ax=axes[2], atlas="SARM5", style="solid",
                       **style)
axes[0].set_title("SARM level-6 lobules", fontsize=9)
axes[1].set_title("values on the lobules (SARM6)", fontsize=9)
axes[2].set_title('values on SARM level 5, style="solid"', fontsize=9)
fig.colorbar(sm, ax=axes[1:], shrink=0.6, label="value")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "macaque.png"), dpi=150)
