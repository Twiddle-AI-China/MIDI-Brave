from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Iterable

import numpy as np


def canonical_anchor(trajectories: np.ndarray) -> np.ndarray:
    """Robust 128D identity anchor over note/render trajectories.

    Input is [observations, frames, 128]. Each trajectory first contributes
    its temporal median; the preset anchor is the median across observations.
    This prevents one long sustain or a single stochastic render dominating.
    """
    value = np.asarray(trajectories, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 128 or value.shape[0] < 1:
        raise ValueError("trajectories must be [observations,frames,128]")
    if not np.isfinite(value).all():
        raise ValueError("trajectories contain non-finite values")
    return np.median(np.median(value, axis=1), axis=0).astype(np.float32)


@dataclass(frozen=True)
class TimbreAtlas:
    preset_ids: tuple[str, ...]
    anchors: np.ndarray
    center: np.ndarray
    components: np.ndarray
    coordinates: np.ndarray
    scale: np.ndarray
    mutual_knn: np.ndarray

    @classmethod
    def fit(
        cls,
        preset_trajectories: dict[str, np.ndarray],
        *,
        dimensions: int = 8,
        neighbors: int = 4,
        fit_presets: Iterable[str] | None = None,
    ) -> "TimbreAtlas":
        if len(preset_trajectories) < dimensions + 1:
            raise ValueError("atlas needs at least dimensions+1 presets")
        preset_ids = tuple(sorted(preset_trajectories))
        anchors = np.stack([canonical_anchor(preset_trajectories[key]) for key in preset_ids])
        fit_set = set(preset_ids if fit_presets is None else fit_presets)
        if not fit_set.issubset(preset_ids) or len(fit_set) < dimensions + 1:
            raise ValueError("fit_presets must select at least dimensions+1 known presets")
        fit_anchors = anchors[[item in fit_set for item in preset_ids]]
        center = fit_anchors.mean(0)
        _, singular, vh = np.linalg.svd(fit_anchors - center, full_matrices=False)
        components = vh[:dimensions].astype(np.float32)
        coordinates = ((anchors - center) @ components.T).astype(np.float32)
        scale = coordinates.std(0, ddof=0).clip(1.0e-5).astype(np.float32)
        normalized = coordinates / scale
        distance = np.linalg.norm(normalized[:, None] - normalized[None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        nearest = np.argsort(distance, axis=1)[:, :neighbors]
        adjacency = np.zeros((len(preset_ids), len(preset_ids)), dtype=np.bool_)
        for row, columns in enumerate(nearest):
            adjacency[row, columns] = True
        mutual = adjacency & adjacency.T
        return cls(
            preset_ids, anchors.astype(np.float32), center.astype(np.float32), components,
            coordinates, scale, mutual,
        )

    def transform(self, anchors: np.ndarray) -> np.ndarray:
        value = np.asarray(anchors, dtype=np.float32)
        if value.shape[-1] != 128:
            raise ValueError("anchors must end in 128D")
        return ((value - self.center) @ self.components.T).astype(np.float32)

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinates, dtype=np.float32)
        if value.shape[-1] != self.components.shape[0]:
            raise ValueError("coordinate dimension does not match atlas")
        return (self.center + value @ self.components).astype(np.float32)

    def project_local(self, query: np.ndarray, neighbors: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Project a query to a stable local convex hull of k legal anchors."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape != (self.components.shape[0],):
            raise ValueError("query has invalid atlas dimension")
        normalized = self.coordinates / self.scale
        query_normalized = query / self.scale
        distance = np.linalg.norm(normalized - query_normalized[None], axis=1)
        indices = np.argsort(distance)[:neighbors]
        local = self.coordinates[indices]
        inverse = 1.0 / np.maximum(distance[indices], 1.0e-6)
        weights = inverse / inverse.sum()
        projected = (weights[:, None] * local).sum(0)
        anchor = (weights[:, None] * self.anchors[indices]).sum(0)
        return projected.astype(np.float32), anchor.astype(np.float32)

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Return deterministic components of the mutual-kNN graph."""
        unseen = set(range(len(self.preset_ids)))
        components: list[tuple[int, ...]] = []
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            stack = [start]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in np.flatnonzero(self.mutual_knn[current]):
                    index = int(neighbor)
                    if index in unseen:
                        unseen.remove(index)
                        stack.append(index)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def project_details(
        self,
        query: np.ndarray,
        *,
        neighbors: int = 4,
        component: int | None = None,
    ) -> dict[str, object]:
        """Project to a local legal hull and expose auditable navigation data."""
        value = np.asarray(query, dtype=np.float32).reshape(-1)
        if value.shape != (self.components.shape[0],) or not np.isfinite(value).all():
            raise ValueError("query has invalid atlas coordinates")
        components = self.connected_components()
        if component is None:
            normalized_all = self.coordinates / self.scale
            nearest = int(np.argmin(np.linalg.norm(
                normalized_all - value[None] / self.scale, axis=1,
            )))
            component = next(
                index for index, nodes in enumerate(components) if nearest in nodes
            )
            candidate_indices = np.asarray(components[component], dtype=np.int64)
        else:
            if component < 0 or component >= len(components):
                raise ValueError("component is out of range")
            candidate_indices = np.asarray(components[component], dtype=np.int64)
        normalized = self.coordinates[candidate_indices] / self.scale
        distance = np.linalg.norm(normalized - value[None] / self.scale, axis=1)
        local_count = min(max(1, int(neighbors)), candidate_indices.size)
        order = np.argsort(distance)[:local_count]
        indices = candidate_indices[order]
        inverse = 1.0 / np.maximum(distance[order], 1.0e-6)
        weights = inverse / inverse.sum()
        projected = (weights[:, None] * self.coordinates[indices]).sum(0)
        anchor = (weights[:, None] * self.anchors[indices]).sum(0)
        membership = {
            node: component_index
            for component_index, nodes in enumerate(components)
            for node in nodes
        }
        return {
            "coordinate": projected.astype(np.float32),
            "anchor": anchor.astype(np.float32),
            "indices": indices.astype(np.int64),
            "preset_ids": tuple(self.preset_ids[int(index)] for index in indices),
            "weights": weights.astype(np.float32),
            "component": int(membership[int(indices[0])]),
        }

    def graph_path(self, source_index: int, target_index: int) -> tuple[int, ...] | None:
        """Shortest normalized-distance path, or ``None`` across components."""
        count = len(self.preset_ids)
        if not (0 <= source_index < count and 0 <= target_index < count):
            raise ValueError("atlas node index is out of range")
        if source_index == target_index:
            return (source_index,)
        coordinates = self.coordinates / self.scale
        queue: list[tuple[float, int]] = [(0.0, source_index)]
        distance = {source_index: 0.0}
        previous: dict[int, int] = {}
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != distance.get(current):
                continue
            if current == target_index:
                path = [current]
                while current in previous:
                    current = previous[current]
                    path.append(current)
                return tuple(reversed(path))
            for raw_neighbor in np.flatnonzero(self.mutual_knn[current]):
                neighbor = int(raw_neighbor)
                edge = float(np.linalg.norm(coordinates[current] - coordinates[neighbor]))
                candidate = cost + edge
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        return None

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                preset_ids=np.asarray(self.preset_ids),
                anchors=self.anchors,
                center=self.center,
                components=self.components,
                coordinates=self.coordinates,
                scale=self.scale,
                mutual_knn=self.mutual_knn,
            )
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        metadata = {
            "schema": "midibrave.atlas-flow.atlas.v1",
            "sha256": digest,
            "presets": len(self.preset_ids),
            "anchor_dim": 128,
            "atlas_dim": int(self.components.shape[0]),
            "mutual_edges": int(self.mutual_knn.sum() // 2),
        }
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimbreAtlas":
        with np.load(path, allow_pickle=False) as value:
            return cls(
                tuple(str(item) for item in value["preset_ids"].tolist()),
                value["anchors"].astype(np.float32),
                value["center"].astype(np.float32),
                value["components"].astype(np.float32),
                value["coordinates"].astype(np.float32),
                value["scale"].astype(np.float32),
                value["mutual_knn"].astype(np.bool_),
            )


def explained_variance_ratio(anchors: np.ndarray, components: np.ndarray) -> float:
    value = np.asarray(anchors, dtype=np.float64)
    centered = value - value.mean(0)
    total = np.square(centered).sum()
    retained = np.square(centered @ np.asarray(components, dtype=np.float64).T).sum()
    return float(retained / max(total, 1.0e-12))


__all__ = ["TimbreAtlas", "canonical_anchor", "explained_variance_ratio"]
