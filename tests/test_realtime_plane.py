import numpy as np
import pytest

from midibrave.realtime_plane import TimbrePlane


def bank() -> np.ndarray:
    rng = np.random.default_rng(20260722)
    value = rng.normal(size=(64, 512)).astype(np.float32)
    value[:, 0] += np.linspace(-4.0, 4.0, 64)
    value[:, 1] += np.sin(np.linspace(0.0, 8.0, 64)) * 2.0
    return value


def test_pca_plane_is_deterministic_normalized_and_bounded():
    seed = bank()
    first = TimbrePlane.fit(seed)
    second = TimbrePlane.fit(bank())
    np.testing.assert_allclose(first.components, second.components)
    np.testing.assert_allclose(first.points, second.points)
    for component in first.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] > 0.0
    normalized = seed.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    np.testing.assert_allclose(first.mean, normalized.mean(axis=0))
    coordinates = (normalized - first.mean) @ first.components.T
    np.testing.assert_allclose(first.low, np.quantile(coordinates, 0.02, axis=0))
    np.testing.assert_allclose(first.high, np.quantile(coordinates, 0.98, axis=0))
    for x, y in [(-1, -1), (0, 0), (1, 1), (-0.4, 0.7)]:
        mapped = first.map_xy(x, y)
        assert mapped.shape == (512,)
        assert mapped.dtype == np.float32
        assert np.isfinite(mapped).all()
        assert np.linalg.norm(mapped) == pytest.approx(1.0, abs=1e-6)
    assert np.max(np.abs(first.points)) <= 1.0


def test_pca_plane_rejects_invalid_seed_banks():
    with pytest.raises(ValueError, match="seed_clap"):
        TimbrePlane.fit(np.zeros((2, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="seed_clap"):
        TimbrePlane.fit(bank()[:, :511])
    broken = bank()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TimbrePlane.fit(broken)
