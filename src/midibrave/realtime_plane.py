from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimbrePlane:
    mean: np.ndarray
    components: np.ndarray
    low: np.ndarray
    high: np.ndarray
    points: np.ndarray

    @classmethod
    def fit(cls, seed_clap: np.ndarray) -> "TimbrePlane":
        values = np.asarray(seed_clap, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 512:
            raise ValueError("seed_clap must have at least three rows and 512 dimensions")
        if not np.isfinite(values).all():
            raise ValueError("seed_clap must be finite")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("seed_clap rows must have nonzero norm")
        values = values / norms
        mean = values.mean(axis=0)
        centered = values - mean
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        if singular.shape[0] < 2 or singular[1] <= 1e-12:
            raise ValueError("seed_clap must span two PCA dimensions")
        components = right[:2].copy()
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0:
                row *= -1
        coordinates = centered @ components.T
        low = np.quantile(coordinates, 0.02, axis=0)
        high = np.quantile(coordinates, 0.98, axis=0)
        span = high - low
        if np.any(span <= 1e-9):
            raise ValueError("seed_clap PCA range is degenerate")
        points = np.clip((coordinates - low) / span * 2.0 - 1.0, -1.0, 1.0)
        return cls(mean, components, low, high, points.astype(np.float32))

    def map_xy(self, x: float, y: float) -> np.ndarray:
        xy = np.clip(np.asarray([x, y], dtype=np.float64), -1.0, 1.0)
        coordinate = self.low + (xy + 1.0) * 0.5 * (self.high - self.low)
        value = self.mean + coordinate @ self.components
        value /= max(float(np.linalg.norm(value)), 1e-12)
        return value.astype(np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "bounds": [-1.0, 1.0],
            "quantiles": [0.02, 0.98],
            "points": self.points.tolist(),
        }
