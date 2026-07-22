from pathlib import Path


PAGE = Path(__file__).parents[1] / "audition" / "scratch-pad-final" / "index.html"


def test_scratch_pad_audition_page_exposes_three_independent_axes():
    html = PAGE.read_text(encoding="utf-8")

    assert 'id="seedSelect"' in html
    assert 'id="timbreSelect"' in html
    assert 'id="noteSelect"' in html
    assert 'id="noteMatrix"' in html
    assert 'id="timbreMatrix"' in html
    assert 'id="seedMatrix"' in html


def test_scratch_pad_audition_page_loads_runtime_manifest_and_audio():
    html = PAGE.read_text(encoding="utf-8")

    assert "runtime-audition-manifest.json" in html
    assert "AudioContext" in html
    assert 'id="waveform"' in html
    assert 'role="status"' in html
    assert "encoder-free" in html


def test_scratch_pad_audition_page_is_utf8_without_known_mojibake():
    html = PAGE.read_text(encoding="utf-8")

    assert "潜空间试听台" in html
    assert "音色" in html
    for mojibake in ("锟", "娼滅", "姝ｅ", "鈫"):
        assert mojibake not in html
