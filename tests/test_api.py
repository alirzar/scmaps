"""The public API: values / colours by position or by name, the colour map and
range, highlights, and the matplotlib helpers. The renderers are replaced by
recorders, so these tests check what is asked for, not what is drawn."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import scmaps                                               # noqa: E402
from scmaps import maps                                     # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    rec = {}

    def sub(rgb, layout, dpi, nan_rgb):
        rec["subcortex"] = dict(rgb=np.asarray(rgb), layout=layout, dpi=dpi, nan_rgb=nan_rgb)
        return np.zeros((4, 4, 4), np.uint8)

    def cer(rgb, alphas, dpi):
        rec["cerebellum"] = dict(rgb=np.asarray(rgb), alphas=np.asarray(alphas), dpi=dpi)
        return np.zeros((4, 4, 4), np.uint8)
    monkeypatch.setattr(maps, "render_subcortex", sub)
    monkeypatch.setattr(maps, "render_cerebellum", cer)
    return rec


def test_region_names():
    assert len(scmaps.SUBCORTEX_REGIONS) == len(scmaps.CEREBELLUM_REGIONS) == 32
    assert scmaps.SUBCORTEX_REGIONS[:2] == ("aHIP-lh", "pHIP-lh")
    assert scmaps.CEREBELLUM_REGIONS[0] == "M1L" and scmaps.CEREBELLUM_REGIONS[-1] == "S5R"
    assert scmaps.SUBCORTEX_LAYOUTS == ("row", "grid", "lateral", "left")


def test_values_through_the_colour_map(calls):
    v = np.linspace(-2, 2, 32)
    v[3] = np.nan
    scmaps.subcortex_map(v, cmap="viridis", vmin=-1, vmax=1, layout="row", dpi=100)
    got = calls["subcortex"]
    cm = matplotlib.colormaps["viridis"]
    ok = ~np.isnan(v)
    np.testing.assert_allclose(got["rgb"][ok], cm(np.clip((v[ok] + 1) / 2, 0, 1))[:, :3])
    assert np.isnan(got["rgb"][3]).all()
    assert (got["layout"], got["dpi"]) == ("row", 100)


def test_default_range_is_the_values_min_and_max(calls):
    v = np.arange(32.0)
    scmaps.cerebellum_map(v, cmap="magma")
    cm = matplotlib.colormaps["magma"]
    np.testing.assert_allclose(calls["cerebellum"]["rgb"][[0, -1]], [cm(0.0)[:3], cm(1.0)[:3]])


def test_values_by_name(calls):
    scmaps.subcortex_map({"aHIP-lh": 1.0, "pCAU-rh": 0.0}, vmin=0, vmax=1)
    rgb = calls["subcortex"]["rgb"]
    cm = matplotlib.colormaps["viridis"]
    np.testing.assert_allclose(rgb[0], cm(1.0)[:3])
    np.testing.assert_allclose(rgb[31], cm(0.0)[:3])
    assert np.isnan(rgb[1:31]).all()


def test_values_as_a_pandas_series(calls):
    pd = pytest.importorskip("pandas")
    s = pd.Series({"M2L": 1.0, "S5R": 0.0})
    scmaps.cerebellum_map(s, vmin=0, vmax=1)
    rgb = calls["cerebellum"]["rgb"]
    assert not np.isnan(rgb[[1, 31]]).any() and np.isnan(rgb[0]).all()


def test_values_in_the_original_numbering_by_name(calls):
    """Data in the atlas's original numbering (label 8 = today's S5L, 9-16 =
    D1L-S4L) land on the right regions when passed by name."""
    v = np.arange(32.0)                             # value = original label - 1
    scmaps.cerebellum_map(dict(zip(scmaps.CEREBELLUM_REGIONS_ORIGINAL, v)), vmin=0, vmax=31)
    rgb = calls["cerebellum"]["rgb"]
    cm = matplotlib.colormaps["viridis"]
    at = scmaps.CEREBELLUM_REGIONS.index
    np.testing.assert_allclose(rgb[at("S5L")], cm(7 / 31)[:3])
    np.testing.assert_allclose(rgb[at("D1L")], cm(8 / 31)[:3])
    np.testing.assert_allclose(rgb[at("S4R")], cm(31 / 31)[:3])
    np.testing.assert_allclose(rgb[at("M1L")], cm(0.0)[:3])


def test_unknown_names_and_wrong_lengths_are_refused(calls):
    with pytest.raises(KeyError):
        scmaps.subcortex_map({"hippocampus": 1.0})
    with pytest.raises(KeyError):
        scmaps.cerebellum_map(np.zeros(32), highlight=["A4L"])    # an original-release name
    with pytest.raises(ValueError):
        scmaps.subcortex_map(np.zeros(31))
    with pytest.raises(ValueError):
        scmaps.subcortex_map(np.zeros(32), colors="red")


def test_colours(calls):
    scmaps.subcortex_map(colors="red")
    np.testing.assert_allclose(calls["subcortex"]["rgb"], np.tile([1.0, 0, 0], (32, 1)))
    scmaps.subcortex_map(colors=[(0.0, 0.0, 1.0)] * 32)
    np.testing.assert_allclose(calls["subcortex"]["rgb"], np.tile([0, 0, 1.0], (32, 1)))
    scmaps.cerebellum_map(colors={"A1L": "#00ff00"})
    rgb = calls["cerebellum"]["rgb"]
    np.testing.assert_allclose(rgb[4], [0, 1.0, 0])
    assert np.isnan(np.delete(rgb, 4, axis=0)).all()
    scmaps.subcortex_map(colors=["red"] * 31 + [None])
    assert np.isnan(calls["subcortex"]["rgb"][31]).all()


def test_highlight_paints_and_makes_the_seed_unshaded(calls):
    scmaps.cerebellum_map(np.zeros(32), highlight=["M1L", "M1R"], alpha=0.6)
    got = calls["cerebellum"]
    np.testing.assert_allclose(got["rgb"][[0, 16]], np.tile([1.0, 215 / 255, 0.0], (2, 1)))
    assert got["alphas"][[0, 16]].tolist() == [0.0, 0.0]
    assert (np.delete(got["alphas"], [0, 16]) == 0.6).all()
    mask = np.zeros(32, bool)
    mask[5] = True
    scmaps.subcortex_map(np.zeros(32), highlight=mask, highlight_color="black")
    np.testing.assert_allclose(calls["subcortex"]["rgb"][5], [0, 0, 0])


def test_nan_colour_is_passed_on(calls):
    scmaps.subcortex_map(nan_color="white")
    assert calls["subcortex"]["nan_rgb"] == (1.0, 1.0, 1.0)


def test_plot_helpers_draw_and_return_a_mappable(calls):
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2)
    sm = scmaps.plot_subcortex(np.arange(32.0), ax=a, cmap="RdBu_r", vmin=-5, vmax=5)
    assert sm.norm.vmin == -5 and sm.norm.vmax == 5 and sm.cmap.name == "RdBu_r"
    sm = scmaps.plot_cerebellum(np.arange(32.0), ax=b)
    assert (sm.norm.vmin, sm.norm.vmax) == (0, 31)
    assert len(a.images) == len(b.images) == 1 and not a.axison
    fig.colorbar(sm, ax=b)
    assert scmaps.plot_cerebellum(colors="red", ax=b) is None
    plt.close(fig)
