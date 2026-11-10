import pytest

pytest.importorskip("matplotlib")

from test_us_damlist import GRID, fake_sources  # noqa: F401  (fixture)

from pytedi.us import damlist as dl
from pytedi.us.plot import plot_candidates


def test_plot_writes_a_png(fake_sources, tmp_path):
    r = dl.build_damlist(GRID)
    out = tmp_path / "map.png"
    fig = plot_candidates(r, out, title="Synthetic valley")
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("NHD" in t for t in labels) and any("NAIP" in t for t in labels)
