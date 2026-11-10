import warnings

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("contextily")

import xyzservices.providers as xyz  # noqa: E402

from test_us_damlist import GRID, fake_sources  # noqa: E402,F401  (fixture)

from pytedi.us import damlist as dl  # noqa: E402
from pytedi.us import plot  # noqa: E402
from pytedi.us.plot import plot_candidates  # noqa: E402


@pytest.fixture
def tiles(monkeypatch):
    """Record basemap requests instead of fetching tiles."""
    calls = []
    monkeypatch.setattr(plot, "_add_basemap", lambda ax, source, crs, zoom: calls.append((source, crs, zoom)))
    return calls


def texts(fig):
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text"))]


def test_plot_writes_a_png(fake_sources, tiles, tmp_path):
    r = dl.build_damlist(GRID)
    out = tmp_path / "map.png"
    fig = plot_candidates(r, out, title="Synthetic valley")
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("NHD" in t for t in labels) and any("NAIP" in t for t in labels)


def test_default_basemap_is_usgs_topo_with_attribution(fake_sources, tiles):
    fig = plot_candidates(dl.build_damlist(GRID))
    (source, crs, zoom), = tiles
    assert source.name == "USGS.USTopo" and crs == "EPSG:5070" and zoom == "auto"
    assert any("USGS The National Map" in t for t in texts(fig))


def test_other_provider_and_zoom(fake_sources, tiles):
    fig = plot_candidates(dl.build_damlist(GRID), basemap=xyz.USGS.USImagery, basemap_zoom=14)
    assert tiles[0][0].name == "USGS.USImagery" and tiles[0][2] == 14
    assert any(xyz.USGS.USImagery.attribution in t for t in texts(fig))


def test_no_basemap(fake_sources, tiles):
    fig = plot_candidates(dl.build_damlist(GRID), basemap=None)
    assert tiles == [] and not any("Basemap" in t for t in texts(fig))


def test_offline_basemap_warns_and_still_draws(fake_sources, monkeypatch, tmp_path):
    def offline(ax, source, crs, zoom):
        raise OSError("no network")

    monkeypatch.setattr(plot, "_add_basemap", offline)
    out = tmp_path / "map.png"
    with pytest.warns(UserWarning, match="basemap"):
        fig = plot_candidates(dl.build_damlist(GRID), out)
    assert out.stat().st_size > 10_000 and not any("Basemap" in t for t in texts(fig))


def test_basemap_must_be_a_tile_provider(fake_sources, tiles):
    with pytest.raises(TypeError, match="xyzservices"):
        plot_candidates(dl.build_damlist(GRID), basemap="USGS")
