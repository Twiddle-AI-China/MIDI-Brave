from pathlib import Path


PAGE = (
    Path(__file__).parents[1]
    / "audition"
    / "zrave-sequence-comparison"
    / "index.html"
)


def test_page_exposes_comparison_and_random_sections() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "audition-manifest.json" in html
    assert 'id="comparisonGroups"' in html
    assert 'id="randomContinuations"' in html
    assert 'data-version="1"' in html
    for category in ("Pad", "Bass", "Lead", "Pluck", "Keys"):
        assert category in html


def test_page_has_synchronized_switching_contract() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "switchVariant" in html
    assert "Original" in html
    assert "Direct reconstruction" in html
    assert "Predicted sequence" in html
    assert "keydown" in html
    assert "currentTime" in html
    assert "pauseOtherPlayers" in html


def test_page_explains_standalone_codec_and_seed_boundary() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "Standalone RAVE" in html
    assert "0 CLAP · 0 MIDI" in html
    assert "Direct vs Predicted" in html
    assert "seed-marker" in html
    assert "transition_seconds" in html
    assert 'role="status"' in html


def test_page_does_not_describe_conditional_decoding() -> None:
    html = PAGE.read_text(encoding="utf-8").casefold()

    assert "pad-focused decoder" not in html
    assert "clap embedding" not in html
    assert "midi conditioning" not in html


def test_page_resolves_generated_assets_without_external_dependencies() -> None:
    html = PAGE.read_text(encoding="utf-8")

    assert "new URL(path, manifestURL)" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
