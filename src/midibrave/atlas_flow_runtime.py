from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch

from .atlas_flow_atlas import TimbreAtlas
from .atlas_flow_config import AtlasFlowConfig, load_atlas_flow_config
from .atlas_flow_data import lifecycle_frames
from .atlas_flow_model import AtlasFlowSystem


def describe_device(device: torch.device) -> str:
    """Human-readable device name, for a runtime that no longer assumes CUDA."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return f"CPU x{torch.get_num_threads()}"


LIVE_SOLVER_STEPS = 4
LIVE_BLOCK_SAMPLES = 4_096
LIVE_TRANSITION_SECONDS = 0.5
DEFAULT_MORPH_SECONDS = 2.0
MIN_MORPH_SECONDS = 0.5
MAX_MORPH_SECONDS = 5.0
SUSTAIN_LOOP_START_SECONDS = 1.2
SUSTAIN_LOOP_END_MARGIN_SECONDS = 0.2
SUSTAIN_LOOP_CROSSFADE_SECONDS = 0.2
RELEASE_CROSSFADE_SECONDS = 0.1
NOTE_CHANGE_CROSSFADE_SECONDS = 0.05
STOP_FADE_SECONDS = 0.15


@dataclass(frozen=True)
class LiveControl:
    seq: int
    pca_normalized: tuple[float, ...]
    note: int
    velocity: float
    temperature: float
    morph_seconds: float


@dataclass(frozen=True)
class PlanRequest:
    revision: int
    control: LiveControl
    seed: int
    source_coordinate: np.ndarray
    source_anchor: np.ndarray
    source_component: int
    restart: bool


@dataclass(frozen=True)
class RenderPlan:
    revision: int
    note: int
    trajectory: torch.Tensor
    anchor_timeline: torch.Tensor
    coordinate_timeline: torch.Tensor
    waveform: np.ndarray
    requested_normalized: np.ndarray
    projected_normalized: np.ndarray
    projected_coordinate: np.ndarray
    projected_anchor: np.ndarray
    component: int
    plan_ms: float
    mode: str
    morph_seconds: float
    restart: bool


@dataclass
class PlaybackLayer:
    plan: RenderPlan
    cursor: int


class AtlasFlowLiveSession:
    """Continuous monophonic voice backed by canonical five-second plans.

    The learned model still predicts its complete trained trajectory. Runtime
    time-warps only the mature sustain region while a key is held and enters
    the learned release after note-off. GPU plans are prepared without holding
    the audio lock, so a slow plan can delay a morph but can never stop PCM.
    """

    def __init__(self, engine: "AtlasFlowLiveEngine") -> None:
        self.engine = engine
        self.lock = threading.RLock()
        default = tuple(float(value) for value in engine.default_normalized)
        self.control = LiveControl(
            0, default, 50, 1.0, 0.0, DEFAULT_MORPH_SECONDS,
        )
        self.seed = 20260822
        self.requested_revision = 0
        self.planned_revision = -1
        self.restart_revision = -1
        self.active: PlaybackLayer | None = None
        self.incoming: PlaybackLayer | None = None
        self.queued: RenderPlan | None = None
        self.fade_cursor = 0
        self.fade_samples = 1
        self.state = "idle"
        self.release_tail: np.ndarray | None = None
        self.release_tail_cursor = 0
        self.stop_cursor = 0
        self.stop_samples = 0
        self.last_plan_ms = 0.0
        self.last_mode = "idle"
        self.last_error: str | None = None

        rate = engine.config.data.sample_rate
        cfg = engine.config.data
        self.loop_start = max(
            cfg.note_on_sample + 1,
            round(SUSTAIN_LOOP_START_SECONDS * rate),
        )
        self.loop_end = min(
            cfg.note_off_sample - 1,
            cfg.note_off_sample - round(SUSTAIN_LOOP_END_MARGIN_SECONDS * rate),
        )
        self.loop_crossfade = round(SUSTAIN_LOOP_CROSSFADE_SECONDS * rate)
        if self.loop_end - self.loop_start <= 2 * self.loop_crossfade:
            raise ValueError("sustain loop is too short for its crossfade")

    def _make_control(
        self,
        seq: int,
        *,
        pca_normalized: object | None,
        x: float | None,
        y: float | None,
        note: int,
        velocity: float,
        temperature: float,
        morph_seconds: float,
    ) -> LiveControl:
        if pca_normalized is None:
            pca = np.asarray(self.control.pca_normalized, dtype=np.float64).copy()
            if x is not None:
                pca[0] = x
            if y is not None:
                pca[1] = y
        else:
            pca = np.asarray(pca_normalized, dtype=np.float64).reshape(-1)
        if pca.shape != (self.engine.atlas.components.shape[0],):
            raise ValueError("pcaNormalized must contain exactly eight values")
        values = np.concatenate((
            np.asarray((seq, note, velocity, temperature, morph_seconds), dtype=np.float64),
            pca,
        ))
        if not np.isfinite(values).all():
            raise ValueError("control values must be finite")
        return LiveControl(
            int(seq), tuple(float(item) for item in np.clip(pca, -1.0, 1.0)),
            int(np.clip(note, 36, 71)), float(np.clip(velocity, 0.0, 1.0)),
            float(np.clip(temperature, 0.0, 1.0)),
            float(np.clip(morph_seconds, MIN_MORPH_SECONDS, MAX_MORPH_SECONDS)),
        )

    def request_start(
        self,
        *,
        note: int,
        velocity: float,
        seed: int,
        temperature: float,
        morph_seconds: float = DEFAULT_MORPH_SECONDS,
        pca_normalized: object | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        with self.lock:
            self.seed = int(seed)
            self.control = self._make_control(
                self.control.seq + 1,
                pca_normalized=pca_normalized, x=x, y=y,
                note=note, velocity=velocity, temperature=temperature,
                morph_seconds=morph_seconds,
            )
            self.requested_revision += 1
            self.restart_revision = self.requested_revision
            self.state = "planning" if self.active is None else "held_sustain"
            self.release_tail = None
            self.stop_samples = 0
            self.last_error = None

    def request_control(
        self,
        *,
        seq: int,
        note: int,
        velocity: float,
        temperature: float,
        morph_seconds: float = DEFAULT_MORPH_SECONDS,
        pca_normalized: object | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> bool:
        with self.lock:
            if seq <= self.control.seq:
                return False
            candidate = self._make_control(
                seq, pca_normalized=pca_normalized, x=x, y=y,
                note=note, velocity=velocity, temperature=temperature,
                morph_seconds=morph_seconds,
            )
            changed_plan = (
                candidate.pca_normalized != self.control.pca_normalized
                or candidate.note != self.control.note
                or candidate.temperature != self.control.temperature
            )
            note_changed = candidate.note != self.control.note
            self.control = candidate
            if self.state == "release":
                return True
            if changed_plan:
                self.requested_revision += 1
                if note_changed:
                    self.restart_revision = self.requested_revision
                if self.state == "idle":
                    self.state = "planning"
            return True

    # Backwards-compatible synchronous entrypoints used by local tests/tools.
    def start(self, **kwargs: object) -> None:
        self.request_start(**kwargs)
        while self.plan_pending():
            self.plan_latest()

    def update(self, **kwargs: object) -> bool:
        accepted = self.request_control(**kwargs)
        while accepted and self.plan_pending():
            self.plan_latest()
        return accepted

    def reseed(self, seed: int) -> None:
        with self.lock:
            self.seed = int(seed)
            if self.state not in {"release", "idle"}:
                self.requested_revision += 1

    def plan_pending(self) -> bool:
        with self.lock:
            return (
                self.state != "release"
                and self.requested_revision > self.planned_revision
            )

    def _audible_source_locked(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.active is None:
            index = self.engine.center_index
            return (
                self.engine.atlas.coordinates[index].copy(),
                self.engine.atlas.anchors[index].copy(),
                self.engine.component_membership[index],
            )
        if self.incoming is None:
            plan = self.active.plan
            return (
                plan.projected_coordinate.copy(), plan.projected_anchor.copy(),
                plan.component,
            )
        progress = float(np.clip(self.fade_cursor / max(self.fade_samples, 1), 0.0, 1.0))
        left = self.active.plan
        right = self.incoming.plan
        coordinate = left.projected_coordinate * (1.0 - progress) + right.projected_coordinate * progress
        anchor = left.projected_anchor * (1.0 - progress) + right.projected_anchor * progress
        component = right.component if progress >= 0.5 else left.component
        return coordinate.astype(np.float32), anchor.astype(np.float32), component

    def _plan_request(self) -> PlanRequest | None:
        with self.lock:
            if not self.plan_pending():
                return None
            coordinate, anchor, component = self._audible_source_locked()
            revision = self.requested_revision
            return PlanRequest(
                revision=revision,
                control=self.control,
                seed=self.seed,
                source_coordinate=coordinate,
                source_anchor=anchor,
                source_component=component,
                restart=(revision == self.restart_revision),
            )

    @torch.no_grad()
    def _compute_plan(self, request: PlanRequest) -> RenderPlan:
        started = time.perf_counter()
        system = self.engine.system
        cfg = self.engine.config
        device = self.engine.device
        requested = np.asarray(request.control.pca_normalized, dtype=np.float32)
        query = self.engine.denormalize(requested)
        target = self.engine.atlas.project_details(query)
        target_coordinate = np.asarray(target["coordinate"], dtype=np.float32)
        target_anchor = np.asarray(target["anchor"], dtype=np.float32)
        target_component = int(target["component"])
        projected_normalized = self.engine.normalize(target_coordinate)
        cross_component = (
            request.source_component >= 0
            and request.source_component != target_component
        )

        total = self.engine.total_trajectory_frames
        context = cfg.model.context_frames
        future = cfg.model.future_frames
        if cross_component or request.restart:
            source_coordinate = target_coordinate
            source_anchor = target_anchor
        else:
            source_coordinate = request.source_coordinate
            source_anchor = request.source_anchor
        history = torch.from_numpy(source_anchor).view(1, 1, -1).expand(1, context, -1).clone().to(device)
        history_anchor = history.clone()
        history_mask = torch.zeros(1, context, dtype=torch.bool, device=device)
        history_mask[:, -1] = True

        transition_frames = max(
            2, round(LIVE_TRANSITION_SECONDS * cfg.data.sample_rate / cfg.data.trajectory_hop)
        )
        source_index = int(np.argmin(np.linalg.norm(
            (self.engine.atlas.coordinates - source_coordinate[None]) / self.engine.atlas.scale,
            axis=1,
        )))
        target_index = int(np.argmin(np.linalg.norm(
            (self.engine.atlas.coordinates - target_coordinate[None]) / self.engine.atlas.scale,
            axis=1,
        )))
        route = None if cross_component else self.engine.atlas.graph_path(source_index, target_index)
        intermediate = () if route is None else route[1:-1]
        route_coordinates = np.stack((
            source_coordinate,
            *(self.engine.atlas.coordinates[index] for index in intermediate),
            target_coordinate,
        )).astype(np.float32)
        route_anchors = np.stack((
            source_anchor,
            *(self.engine.atlas.anchors[index] for index in intermediate),
            target_anchor,
        )).astype(np.float32)
        route_edges = np.linalg.norm(
            np.diff(route_coordinates / self.engine.atlas.scale, axis=0), axis=1,
        )
        if not np.any(route_edges > 1.0e-8):
            route_edges = np.ones_like(route_edges)
        route_cumulative = np.concatenate(([0.0], np.cumsum(route_edges)))
        route_cumulative /= max(float(route_cumulative[-1]), 1.0e-8)

        def navigation(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            phase = np.clip(indices / max(transition_frames - 1, 1), 0.0, 1.0)
            progress = phase * phase * (3.0 - 2.0 * phase)
            coordinates: list[np.ndarray] = []
            anchors: list[np.ndarray] = []
            for item in progress:
                edge = min(
                    len(route_edges) - 1,
                    max(0, int(np.searchsorted(route_cumulative, item, side="right") - 1)),
                )
                width = max(float(route_cumulative[edge + 1] - route_cumulative[edge]), 1.0e-8)
                alpha = float((item - route_cumulative[edge]) / width)
                coordinates.append(route_coordinates[edge] * (1.0 - alpha) + route_coordinates[edge + 1] * alpha)
                anchors.append(route_anchors[edge] * (1.0 - alpha) + route_anchors[edge + 1] * alpha)
            return np.stack(anchors).astype(np.float32), np.stack(coordinates).astype(np.float32)

        chunks: list[torch.Tensor] = []
        anchor_chunks: list[torch.Tensor] = []
        coordinate_chunks: list[torch.Tensor] = []
        produced = 0
        while produced < total:
            local = np.arange(future, dtype=np.float32) + produced
            anchors, coordinates = navigation(local)
            anchor_path = torch.from_numpy(anchors)[None].to(device)
            atlas_path = torch.from_numpy(coordinates)[None].to(device)
            lifecycle = torch.from_numpy(lifecycle_frames(produced, future, cfg.data))[None].to(device)
            generated = system.flow.sample(
                history, history_anchor, anchor_path, atlas_path, lifecycle,
                seed=request.seed + produced,
                temperature=request.control.temperature,
                history_mask=history_mask,
                solver_steps=LIVE_SOLVER_STEPS,
            )
            chunks.append(generated[0].cpu())
            anchor_chunks.append(anchor_path[0].cpu())
            coordinate_chunks.append(atlas_path[0].cpu())
            history = torch.cat((history, generated), 1)[:, -context:]
            history_anchor = torch.cat((history_anchor, anchor_path), 1)[:, -context:]
            history_mask = torch.ones_like(history_mask)
            produced += future

        trajectory = torch.cat(chunks, 0)[:total]
        anchor_timeline = torch.cat(anchor_chunks, 0)[:total]
        coordinate_timeline = torch.cat(coordinate_chunks, 0)[:total]
        audio, _ = system.instrument.decode(
            trajectory.T.contiguous()[None].to(device),
            torch.tensor([request.control.note], device=device),
            cfg.data.render_samples,
            window_start=torch.zeros(1, dtype=torch.long, device=device),
        )
        mono = audio[0, 0].float().cpu().numpy()
        waveform = np.repeat(mono[:, None], 2, axis=1).astype("<f4", copy=False)
        if not np.isfinite(waveform).all() or not bool(torch.isfinite(trajectory).all()):
            raise RuntimeError("Atlas Flow runtime produced non-finite output")
        return RenderPlan(
            revision=request.revision,
            note=request.control.note,
            trajectory=trajectory,
            anchor_timeline=anchor_timeline,
            coordinate_timeline=coordinate_timeline,
            waveform=waveform,
            requested_normalized=requested,
            projected_normalized=projected_normalized,
            projected_coordinate=target_coordinate,
            projected_anchor=target_anchor,
            component=target_component,
            plan_ms=(time.perf_counter() - started) * 1000.0,
            mode=("cross_component_plan" if cross_component else "graph_plan"),
            morph_seconds=request.control.morph_seconds,
            restart=request.restart,
        )

    def _start_incoming_locked(self, plan: RenderPlan) -> None:
        if self.active is None:
            self.active = PlaybackLayer(plan, self.engine.config.data.note_on_sample)
            self.state = "held_sustain"
            return
        note_change = plan.restart or plan.note != self.active.plan.note
        cursor = self.engine.config.data.note_on_sample if note_change else self.active.cursor
        self.incoming = PlaybackLayer(plan, cursor)
        seconds = NOTE_CHANGE_CROSSFADE_SECONDS if note_change else plan.morph_seconds
        self.fade_samples = max(1, round(seconds * self.engine.config.data.sample_rate))
        self.fade_cursor = 0

    def _commit_plan(self, plan: RenderPlan) -> bool:
        with self.lock:
            if plan.revision != self.requested_revision or self.state == "release":
                return False
            self.planned_revision = plan.revision
            self.last_error = None
            if self.incoming is not None:
                self.queued = plan
            else:
                self._start_incoming_locked(plan)
            return True

    def plan_latest(self) -> bool:
        request = self._plan_request()
        if request is None:
            return False
        try:
            plan = self._compute_plan(request)
            # Record the cost even when the plan is about to be thrown away: a
            # roam supersedes most plans before they commit, and reporting only
            # committed ones froze this readout on the cold first plan.
            with self.lock:
                self.last_plan_ms = plan.plan_ms
                self.last_mode = plan.mode
        except Exception as error:
            with self.lock:
                self.last_error = str(error).strip() or type(error).__name__
                if request.revision == self.requested_revision:
                    self.planned_revision = request.revision
            raise
        return self._commit_plan(plan)

    def _read_linear_locked(self, layer: PlaybackLayer, count: int) -> np.ndarray:
        output = np.zeros((count, 2), dtype=np.float32)
        stop = min(layer.cursor + count, layer.plan.waveform.shape[0])
        available = max(0, stop - layer.cursor)
        if available:
            output[:available] = layer.plan.waveform[layer.cursor:stop]
            layer.cursor = stop
        return output

    def _read_held_locked(self, layer: PlaybackLayer, count: int) -> np.ndarray:
        output = np.zeros((count, 2), dtype=np.float32)
        written = 0
        fade_start = self.loop_end - self.loop_crossfade
        while written < count:
            if layer.cursor >= self.loop_end:
                layer.cursor = self.loop_start + self.loop_crossfade
            if layer.cursor < fade_start:
                amount = min(count - written, fade_start - layer.cursor)
                output[written:written + amount] = layer.plan.waveform[layer.cursor:layer.cursor + amount]
                layer.cursor += amount
                written += amount
                continue
            amount = min(count - written, self.loop_end - layer.cursor)
            offset = layer.cursor - fade_start
            phase = np.linspace(
                offset / self.loop_crossfade,
                (offset + amount - 1) / self.loop_crossfade,
                amount,
                dtype=np.float32,
            )[:, None]
            phase = np.clip(phase, 0.0, 1.0) * (math.pi / 2.0)
            old = layer.plan.waveform[layer.cursor:layer.cursor + amount]
            new_start = self.loop_start + offset
            new = layer.plan.waveform[new_start:new_start + amount]
            output[written:written + amount] = old * np.cos(phase) + new * np.sin(phase)
            layer.cursor += amount
            written += amount
        return output

    def _read_layer_locked(self, layer: PlaybackLayer, count: int) -> np.ndarray:
        return (
            self._read_linear_locked(layer, count)
            if self.state == "release"
            else self._read_held_locked(layer, count)
        )

    def _render_mix_locked(self, count: int) -> np.ndarray:
        if self.active is None:
            return np.zeros((count, 2), dtype=np.float32)
        left = self._read_layer_locked(self.active, count)
        if self.incoming is None:
            return left
        right = self._read_layer_locked(self.incoming, count)
        positions = self.fade_cursor + np.arange(count, dtype=np.float32)
        phase = np.clip(positions / max(self.fade_samples - 1, 1), 0.0, 1.0)[:, None]
        phase *= math.pi / 2.0
        mixed = left * np.cos(phase) + right * np.sin(phase)
        self.fade_cursor += count
        if self.fade_cursor >= self.fade_samples:
            self.active = self.incoming
            self.incoming = None
            self.fade_cursor = 0
            if self.queued is not None and self.state != "release":
                queued = self.queued
                self.queued = None
                self._start_incoming_locked(queued)
        return mixed

    def note_off(self) -> None:
        with self.lock:
            if self.active is None or self.state in {"idle", "release"}:
                return
            count = round(RELEASE_CROSSFADE_SECONDS * self.engine.config.data.sample_rate)
            self.release_tail = self._render_mix_locked(count)
            self.release_tail_cursor = 0
            self.state = "release"
            self.queued = None
            self.active.cursor = self.engine.config.data.note_off_sample
            if self.incoming is not None:
                self.incoming.cursor = self.engine.config.data.note_off_sample

    def stop(self) -> None:
        with self.lock:
            if self.active is None:
                self.state = "idle"
                return
            self.stop_cursor = 0
            self.stop_samples = max(
                1, round(STOP_FADE_SECONDS * self.engine.config.data.sample_rate),
            )

    def render_block(self) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        with self.lock:
            block = self._render_mix_locked(LIVE_BLOCK_SAMPLES)
            if self.release_tail is not None:
                remaining = self.release_tail.shape[0] - self.release_tail_cursor
                amount = min(LIVE_BLOCK_SAMPLES, remaining)
                if amount > 0:
                    positions = self.release_tail_cursor + np.arange(amount, dtype=np.float32)
                    phase = positions / max(self.release_tail.shape[0] - 1, 1) * (math.pi / 2.0)
                    old = self.release_tail[self.release_tail_cursor:self.release_tail_cursor + amount]
                    block[:amount] = old * np.cos(phase[:, None]) + block[:amount] * np.sin(phase[:, None])
                    self.release_tail_cursor += amount
                if self.release_tail_cursor >= self.release_tail.shape[0]:
                    self.release_tail = None
            block *= self.control.velocity
            if self.stop_samples:
                positions = self.stop_cursor + np.arange(LIVE_BLOCK_SAMPLES, dtype=np.float32)
                gain = np.clip(1.0 - positions / max(self.stop_samples - 1, 1), 0.0, 1.0)
                block *= gain[:, None]
                self.stop_cursor += LIVE_BLOCK_SAMPLES
                if self.stop_cursor >= self.stop_samples:
                    self.active = self.incoming = None
                    self.queued = None
                    self.stop_samples = 0
                    self.state = "idle"
            if (
                self.state == "release"
                and self.active is not None
                and self.active.cursor >= self.engine.config.data.render_samples
                and (self.incoming is None or self.incoming.cursor >= self.engine.config.data.render_samples)
            ):
                self.active = self.incoming = None
                self.state = "idle"
            return block.astype("<f4", copy=False), (time.perf_counter() - started) * 1000.0

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            target = np.asarray(self.control.pca_normalized, dtype=np.float32)
            if self.active is None:
                audible = self.engine.default_normalized.copy()
                projected = audible.copy()
                component = -1
                note = self.control.note
                cursor = 0
            else:
                note = self.active.plan.note
                cursor = self.active.cursor
                component = self.active.plan.component
                projected = self.active.plan.projected_normalized.copy()
                audible = projected.copy()
                if self.incoming is not None:
                    progress = float(np.clip(
                        self.fade_cursor / max(self.fade_samples, 1), 0.0, 1.0,
                    ))
                    audible = (
                        self.active.plan.projected_normalized * (1.0 - progress)
                        + self.incoming.plan.projected_normalized * progress
                    )
                    component = self.incoming.plan.component if progress >= 0.5 else component
            lifecycle = self.state
            if self.state == "held_sustain" and cursor < self.loop_start:
                lifecycle = "attack"
            return {
                "seq": self.control.seq,
                "note": note,
                "velocity": self.control.velocity,
                "temperature": self.control.temperature,
                "morphSeconds": self.control.morph_seconds,
                "component": component,
                "targetPcaNormalized": target.astype(float).tolist(),
                "projectedPcaNormalized": projected.astype(float).tolist(),
                "audiblePcaNormalized": audible.astype(float).tolist(),
                "lifecycle": lifecycle,
                "sampleCursor": cursor,
                "planMs": self.last_plan_ms,
                "planMode": self.last_mode,
                "planPending": self.requested_revision > self.planned_revision,
                "lastError": self.last_error,
            }


class AtlasFlowLiveEngine:
    def __init__(
        self,
        system: AtlasFlowSystem,
        atlas: TimbreAtlas,
        config: AtlasFlowConfig,
        device: torch.device,
        checkpoint: Path,
    ) -> None:
        self.system = system
        self.atlas = atlas
        self.config = config
        self.device = device
        self.checkpoint = checkpoint
        self.total_trajectory_frames = math.ceil(
            config.data.render_samples / config.data.trajectory_hop
        )
        center_distance = np.linalg.norm(atlas.coordinates / atlas.scale, axis=1)
        self.center_index = int(np.argmin(center_distance))
        self.low = atlas.coordinates.min(0).astype(np.float32)
        self.high = atlas.coordinates.max(0).astype(np.float32)
        self.span = np.maximum(self.high - self.low, 1.0e-8).astype(np.float32)
        self.default_normalized = self.normalize(atlas.coordinates[self.center_index])
        components = atlas.connected_components()
        self.component_membership = {
            node: component
            for component, nodes in enumerate(components)
            for node in nodes
        }

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        checkpoint: str | Path,
        device: str = "cuda:0",
    ) -> "AtlasFlowLiveEngine":
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda was requested but no CUDA device is present")
        if target.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps was requested but no Metal device is present")
        if target.type not in {"cuda", "mps", "cpu"}:
            raise RuntimeError(f"unsupported device: {device}")
        config = load_atlas_flow_config(config_path)
        system = AtlasFlowSystem(config.model, config.data).to(target)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        system.load_state_dict(payload["system"], strict=True)
        system.eval()
        atlas = TimbreAtlas.load(config.data.atlas_path)
        return cls(system, atlas, config, target, Path(checkpoint))

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
        if normalized.shape != (self.atlas.components.shape[0],):
            raise ValueError("normalized PCA coordinates have invalid shape")
        return (self.low + (normalized + 1.0) * 0.5 * self.span).astype(np.float32)

    def normalize(self, coordinates: np.ndarray) -> np.ndarray:
        value = np.asarray(coordinates, dtype=np.float32)
        if value.shape != (self.atlas.components.shape[0],):
            raise ValueError("PCA coordinates have invalid shape")
        return np.clip((value - self.low) / self.span * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    # Legacy two-dimensional helper retained for older clients/tests.
    def query_from_xy(self, x: float, y: float, base: np.ndarray) -> np.ndarray:
        result = np.asarray(base, dtype=np.float32).copy()
        normalized = self.normalize(result)
        normalized[:2] = np.clip(np.asarray((x, y), dtype=np.float32), -1.0, 1.0)
        return self.denormalize(normalized)

    def new_session(self) -> AtlasFlowLiveSession:
        return AtlasFlowLiveSession(self)

    def status(self) -> dict[str, object]:
        components = self.atlas.connected_components()
        normalized = np.stack([
            self.normalize(value) for value in self.atlas.coordinates
        ])
        return {
            "ok": True,
            "engine": "midibrave-atlas-flow-pad-live-v2",
            "classId": "pad",
            "cuda": describe_device(self.device),
            "sampleRate": self.config.data.sample_rate,
            "blockSamples": LIVE_BLOCK_SAMPLES,
            "blockDeadlineMs": LIVE_BLOCK_SAMPLES / self.config.data.sample_rate * 1000.0,
            "solverSteps": LIVE_SOLVER_STEPS,
            "contextFrames": self.config.model.context_frames,
            "futureFrames": self.config.model.future_frames,
            "transitionSeconds": LIVE_TRANSITION_SECONDS,
            "defaultMorphSeconds": DEFAULT_MORPH_SECONDS,
            "morphRangeSeconds": [MIN_MORPH_SECONDS, MAX_MORPH_SECONDS],
            "checkpoint": self.checkpoint.name,
            "defaultPcaNormalized": self.default_normalized.astype(float).tolist(),
            "pcaAxes": [
                {
                    "index": index + 1,
                    "label": f"PC{index + 1}",
                    "minimum": float(self.low[index]),
                    "maximum": float(self.high[index]),
                }
                for index in range(self.atlas.components.shape[0])
            ],
            "points": [
                {
                    "presetId": preset,
                    "x": float(normalized[index, 0]),
                    "y": float(normalized[index, 1]),
                    "pcaNormalized": normalized[index].astype(float).tolist(),
                    "component": int(self.component_membership[index]),
                }
                for index, preset in enumerate(self.atlas.preset_ids)
            ],
            "edges": [
                [int(left), int(right)]
                for left in range(len(self.atlas.preset_ids))
                for right in np.flatnonzero(self.atlas.mutual_knn[left])
                if left < int(right)
            ],
            "components": [len(value) for value in components],
            "limitations": [
                "Pad-only monophonic research demo.",
                "Held sustain is a runtime time-warp of the trained five-second trajectory.",
                "The trained note range is MIDI 36-71 and velocity is output gain only.",
                "All eight PCA controls are projected to a legal local atlas hull.",
                "Disconnected atlas regions use an audio crossfade, not latent interpolation.",
            ],
        }


__all__ = [
    "AtlasFlowLiveEngine", "AtlasFlowLiveSession", "DEFAULT_MORPH_SECONDS",
    "LIVE_BLOCK_SAMPLES", "LIVE_SOLVER_STEPS", "LIVE_TRANSITION_SECONDS",
    "LiveControl", "MAX_MORPH_SECONDS", "MIN_MORPH_SECONDS", "RenderPlan",
]
