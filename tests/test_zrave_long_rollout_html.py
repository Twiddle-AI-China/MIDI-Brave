from pathlib import Path


PAGE = (
    Path(__file__).parents[1]
    / "audition"
    / "zrave-long-rollout"
    / "index.html"
)


def test_page_loads_long_rollout_manifest_and_five_classes() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "long-rollout-manifest.json" in html
    assert 'id="rolloutGrid"' in html
    assert 'data-version="1"' in html
    for category in ("Pad", "Bass", "Lead", "Pluck", "Keys"):
        assert category in html


def test_page_exposes_real_seed_boundary_and_long_jumps() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "32 REAL FRAMES" in html
    assert "320 PREDICTED FRAMES" in html
    assert "createTapeRail" in html
    assert "manifest.rollout_calls" in html
    assert "seed_boundary_seconds" in html
    assert "jumpToFraction" in html
    for label in ("预测起点", "预测 25%", "预测 50%", "预测 75%"):
        assert label in html


def test_page_has_local_accessible_playback_contract() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "keydown" in html
    assert "currentTime" in html
    assert "pauseOtherPlayers" in html
    assert "prefers-reduced-motion" in html
    assert "new URL(relative, manifestURL)" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
