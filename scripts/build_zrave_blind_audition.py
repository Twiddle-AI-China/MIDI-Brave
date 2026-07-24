from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Z-RAVE 盲听台</title>
  <style>
    :root {
      color-scheme: dark;
      --night: #101224;
      --deck: #1a1d34;
      --deck-raised: #242844;
      --line: #414765;
      --paper: #f2f4ff;
      --muted: #aeb5d2;
      --tape: #8bbcff;
      --splice: #ff8e72;
      --safe: #8be0c4;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      min-height: 100vh;
      margin: 0;
      color: var(--paper);
      background:
        radial-gradient(circle at 12% -10%, #343b68 0, transparent 34rem),
        linear-gradient(135deg, transparent 0 49.7%, #191c31 50% 50.3%,
          transparent 50.6%),
        var(--night);
    }

    button,
    input {
      font: inherit;
    }

    button {
      color: inherit;
    }

    button:focus-visible,
    input:focus-visible {
      outline: 3px solid var(--paper);
      outline-offset: 3px;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 80px;
    }

    .masthead {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 28px;
      align-items: end;
      padding-bottom: 28px;
      border-bottom: 4px solid var(--tape);
    }

    .kicker,
    .utility {
      font-family: "Cascadia Code", "SFMono-Regular", monospace;
      font-size: 11px;
      letter-spacing: 0.13em;
      text-transform: uppercase;
    }

    .kicker {
      margin: 0 0 10px;
      color: var(--tape);
    }

    h1 {
      max-width: 820px;
      margin: 0;
      font-family: "Bahnschrift SemiCondensed", "Arial Narrow", sans-serif;
      font-size: clamp(48px, 10vw, 116px);
      font-stretch: condensed;
      font-weight: 800;
      letter-spacing: -0.065em;
      line-height: 0.8;
      text-transform: uppercase;
    }

    .intro {
      max-width: 470px;
      margin: 20px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.75;
    }

    .seal {
      width: 170px;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: var(--night);
      background:
        radial-gradient(circle, var(--night) 0 9%, transparent 10% 27%,
          var(--night) 28% 30%, transparent 31%),
        var(--splice);
      box-shadow: inset 0 0 0 9px var(--night), inset 0 0 0 10px var(--splice);
      font-family: "Cascadia Code", monospace;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-align: center;
      text-transform: uppercase;
      transform: rotate(7deg);
    }

    .control-strip {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: center;
      margin: 26px 0 18px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      background: rgba(26, 29, 52, 0.88);
    }

    .status {
      margin: 0;
      color: var(--muted);
    }

    .status[data-state="ready"] {
      color: var(--safe);
    }

    .status[data-state="error"] {
      color: var(--splice);
    }

    .reveal {
      min-height: 42px;
      padding: 0 16px;
      border: 1px solid var(--splice);
      color: var(--splice);
      background: transparent;
      cursor: pointer;
      font-weight: 750;
    }

    .reveal:hover {
      color: var(--night);
      background: var(--splice);
    }

    .reveal:disabled {
      cursor: default;
      opacity: 0.65;
    }

    .answer-banner {
      display: none;
      margin: 0 0 18px;
      padding: 14px 16px;
      border-left: 5px solid var(--splice);
      color: var(--paper);
      background: #33243a;
    }

    .answer-banner[data-visible="true"] {
      display: block;
    }

    .deck-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: baseline;
      margin: 38px 0 14px;
    }

    .deck-head h2 {
      margin: 0;
      font-family: "Bahnschrift SemiCondensed", "Arial Narrow", sans-serif;
      font-size: 32px;
      letter-spacing: -0.035em;
    }

    .deck-head p {
      margin: 0;
      color: var(--muted);
    }

    .trials {
      display: grid;
      gap: 12px;
    }

    .trial {
      position: relative;
      display: grid;
      grid-template-columns: 148px minmax(0, 1fr);
      min-height: 176px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: var(--deck);
    }

    .trial[data-active="true"] {
      border-color: var(--tape);
      box-shadow: 0 0 0 1px var(--tape);
    }

    .trial-label {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 18px;
      border-right: 1px solid var(--line);
      background: var(--deck-raised);
    }

    .trial-label strong {
      font-family: "Bahnschrift SemiCondensed", "Arial Narrow", sans-serif;
      font-size: 27px;
      letter-spacing: -0.04em;
    }

    .trial-label span {
      color: var(--muted);
    }

    .trial-main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      padding: 16px 18px 14px;
    }

    .variants {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }

    .variant {
      min-height: 54px;
      border: 1px solid var(--line);
      background: transparent;
      cursor: pointer;
      text-align: left;
    }

    .variant:hover {
      border-color: var(--tape);
    }

    .variant[aria-pressed="true"] {
      color: var(--night);
      border-color: var(--tape);
      background: var(--tape);
      font-weight: 800;
    }

    .variant b {
      display: inline-block;
      width: 28px;
      font-family: "Cascadia Code", monospace;
    }

    .transport {
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr) 112px;
      gap: 12px;
      align-items: center;
      margin-top: 15px;
    }

    .play {
      width: 52px;
      height: 44px;
      border: 1px solid var(--splice);
      color: var(--splice);
      background: transparent;
      cursor: pointer;
      font-size: 18px;
    }

    .play:hover {
      color: var(--night);
      background: var(--splice);
    }

    .seek {
      width: 100%;
      accent-color: var(--tape);
    }

    .time {
      color: var(--muted);
      font-family: "Cascadia Code", monospace;
      font-size: 10px;
      text-align: right;
    }

    .tape-rail {
      position: relative;
      height: 10px;
      margin-top: 13px;
      overflow: hidden;
      border: 1px solid #555c7a;
      background: repeating-linear-gradient(
        90deg,
        #303653 0 20px,
        #272c47 20px 40px
      );
    }

    .tape-progress {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0;
      background: var(--tape);
    }

    .trial-answer {
      display: none;
      position: absolute;
      top: 0;
      right: 0;
      padding: 6px 9px;
      color: var(--night);
      background: var(--splice);
      font-family: "Cascadia Code", monospace;
      font-size: 10px;
      font-weight: 800;
    }

    .trial-answer[data-visible="true"] {
      display: block;
    }

    .notes {
      margin-top: 26px;
      padding: 18px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: rgba(26, 29, 52, 0.72);
      font-size: 13px;
      line-height: 1.7;
    }

    .notes strong {
      color: var(--paper);
    }

    @media (max-width: 720px) {
      .masthead,
      .control-strip,
      .deck-head {
        grid-template-columns: 1fr;
      }

      .seal {
        display: none;
      }

      .trial {
        grid-template-columns: 1fr;
      }

      .trial-label {
        min-height: 72px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
        flex-direction: row;
        align-items: end;
      }
    }

    @media (max-width: 500px) {
      .shell {
        width: min(100% - 18px, 1180px);
        padding-top: 18px;
      }

      .variants,
      .transport {
        grid-template-columns: 1fr;
      }

      .play {
        width: 100%;
      }

      .time {
        text-align: left;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        scroll-behavior: auto !important;
        transition: none !important;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <div>
        <p class="kicker">Standalone RAVE / Transformer continuation</p>
        <h1>Blind roll<br>tape test</h1>
        <p class="intro">
          每行是同一段真实 latent 的未来。先听 Direct，再在 A 与 B 之间切换；
          切换时播放头位置不变。A/B 身份在你点击揭晓前不会加载。
        </p>
      </div>
      <div class="seal" aria-hidden="true">identity<br>sealed</div>
    </header>

    <section class="control-strip" aria-label="盲听状态">
      <p class="status utility" id="status" data-state="loading"
        role="status" aria-live="polite">正在装载盲听带…</p>
      <button class="reveal" id="revealButton" type="button" disabled>
        揭晓 A / B
      </button>
    </section>

    <p class="answer-banner" id="answerBanner" data-visible="false"
      aria-live="polite"></p>

    <section aria-labelledby="trialsTitle">
      <div class="deck-head">
        <h2 id="trialsTitle">Matched continuations</h2>
        <p class="utility">D = direct reconstruction · A/B = masked predictors</p>
      </div>
      <div class="trials" id="trials"></div>
    </section>

    <aside class="notes">
      <strong>建议顺序：</strong>先用 D 确认 codec 自身的重建上限；再反复切 A/B，
      特别留意后半段是否塌缩、失去谐波或突然改变响度。键盘 D、A、B 切换当前行，
      空格播放/暂停。页面只读取本地文件，不上传音频。
    </aside>
  </main>

  <script>
    const manifestURL = new URL("blind-manifest.json", window.location.href);
    const answerURL = new URL("blind-answer-key.json", window.location.href);
    const players = new Set();
    const cards = new Map();
    const labels = { direct: "Direct", A: "Version A", B: "Version B" };
    let manifest = null;
    let activeCard = null;

    function assetURL(relative) {
      return new URL(relative, manifestURL).href;
    }

    function formatTime(value) {
      if (!Number.isFinite(value)) return "0:00.000";
      const minutes = Math.floor(value / 60);
      const seconds = (value - minutes * 60).toFixed(3).padStart(6, "0");
      return `${minutes}:${seconds}`;
    }

    function pauseOthers(exception) {
      players.forEach((player) => {
        if (player !== exception && !player.paused) player.pause();
      });
    }

    function activate(card) {
      document.querySelectorAll(".trial").forEach((item) => {
        item.dataset.active = String(item === card);
      });
      activeCard = card;
    }

    function switchVariant(card, variant) {
      const player = card.querySelector("audio");
      const wasPlaying = !player.paused;
      const position = player.currentTime || 0;
      card.dataset.variant = variant;
      card.querySelectorAll(".variant").forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.variant === variant)
        );
      });
      player.src = assetURL(card._row.audio[variant]);
      player.currentTime = position;
      card.querySelector(".active-label").textContent = labels[variant];
      activate(card);
      if (wasPlaying) {
        pauseOthers(player);
        player.play().catch(() => {});
      }
    }

    function buildCard(row, index) {
      const card = document.createElement("article");
      card.className = "trial";
      card.dataset.active = "false";
      card.dataset.variant = "direct";
      card._row = row;
      card.innerHTML = `
        <div class="trial-label">
          <strong>${String(index + 1).padStart(2, "0")}</strong>
          <span class="utility">${row.category || "Unknown"} / ${row.split || "held-out"}</span>
        </div>
        <div class="trial-main">
          <div class="variants" role="group" aria-label="试听版本"></div>
          <div class="transport">
            <button class="play" type="button" aria-label="播放或暂停">▶</button>
            <input class="seek" type="range" min="0" max="1" step="0.001"
              value="0" aria-label="播放位置">
            <span class="time">0:00.000 / 0:00.000</span>
          </div>
          <div>
            <div class="tape-rail" aria-hidden="true">
              <span class="tape-progress"></span>
            </div>
            <span class="active-label utility">Direct</span>
          </div>
        </div>
        <span class="trial-answer"></span>
        <audio preload="metadata"></audio>
      `;

      const switcher = card.querySelector(".variants");
      ["direct", "A", "B"].forEach((variant) => {
        const button = document.createElement("button");
        button.className = "variant";
        button.type = "button";
        button.dataset.variant = variant;
        button.setAttribute("aria-pressed", String(variant === "direct"));
        const key = variant === "direct" ? "D" : variant;
        button.innerHTML = `<b>${key}</b>${labels[variant]}`;
        button.addEventListener("click", () => switchVariant(card, variant));
        switcher.appendChild(button);
      });

      const player = card.querySelector("audio");
      const play = card.querySelector(".play");
      const seek = card.querySelector(".seek");
      const time = card.querySelector(".time");
      const progress = card.querySelector(".tape-progress");
      player.src = assetURL(row.audio.direct);
      players.add(player);

      play.addEventListener("click", () => {
        activate(card);
        if (player.paused) {
          pauseOthers(player);
          player.play().catch(() => {});
        } else {
          player.pause();
        }
      });
      player.addEventListener("play", () => {
        play.textContent = "Ⅱ";
        pauseOthers(player);
        activate(card);
      });
      player.addEventListener("pause", () => {
        play.textContent = "▶";
      });
      player.addEventListener("timeupdate", () => {
        const duration = Number.isFinite(player.duration) ? player.duration : 0;
        const ratio = duration ? player.currentTime / duration : 0;
        seek.value = String(ratio);
        progress.style.width = `${ratio * 100}%`;
        time.textContent = `${formatTime(player.currentTime)} / ${formatTime(duration)}`;
      });
      seek.addEventListener("input", () => {
        if (Number.isFinite(player.duration)) {
          player.currentTime = Number(seek.value) * player.duration;
        }
      });
      card.addEventListener("pointerdown", () => activate(card));
      cards.set(row.id, card);
      return card;
    }

    async function revealAnswers() {
      const button = document.getElementById("revealButton");
      button.disabled = true;
      try {
        const response = await fetch(answerURL, { cache: "no-store" });
        if (!response.ok) throw new Error(`answer HTTP ${response.status}`);
        const key = await response.json();
        key.rows.forEach((row) => {
          const card = cards.get(row.id);
          if (!card) return;
          const marker = card.querySelector(".trial-answer");
          const candidate = row.assignment.A === "candidate" ? "A" : "B";
          marker.textContent = `新模型 = ${candidate}`;
          marker.dataset.visible = "true";
        });
        const banner = document.getElementById("answerBanner");
        banner.textContent = "已揭晓：每行右上角标出了新模型；另一路为基线模型。";
        banner.dataset.visible = "true";
        button.textContent = "身份已揭晓";
      } catch (error) {
        button.disabled = false;
        document.getElementById("status").textContent =
          `答案加载失败：${error.message}`;
      }
    }

    document.getElementById("revealButton")
      .addEventListener("click", revealAnswers);

    document.addEventListener("keydown", (event) => {
      if (!activeCard || event.repeat) return;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
        return;
      }
      const key = event.key.toUpperCase();
      if (["D", "A", "B"].includes(key)) {
        event.preventDefault();
        switchVariant(activeCard, key === "D" ? "direct" : key);
      } else if (event.code === "Space") {
        event.preventDefault();
        activeCard.querySelector(".play").click();
      }
    });

    async function boot() {
      const status = document.getElementById("status");
      try {
        const response = await fetch(manifestURL, { cache: "no-store" });
        if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
        manifest = await response.json();
        if (manifest.schema !== 1 || !Array.isArray(manifest.rows)) {
          throw new Error("不支持的盲听清单");
        }
        const root = document.getElementById("trials");
        manifest.rows.forEach((row, index) => {
          root.appendChild(buildCard(row, index));
        });
        activeCard = root.querySelector(".trial");
        if (activeCard) activate(activeCard);
        status.textContent = `已装载 ${manifest.rows.length} 组匹配试听 · 身份仍封存`;
        status.dataset.state = "ready";
        document.getElementById("revealButton").disabled = false;
      } catch (error) {
        status.textContent = `加载失败：${error.message}。请通过本地 HTTP server 打开。`;
        status.dataset.state = "error";
      }
    }

    boot();
  </script>
</body>
</html>
"""


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError(f"unsupported audition manifest: {path}")
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(f"audition manifest has no comparisons: {path}")
    return value


def _comparison_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in manifest["comparisons"]:
        if not isinstance(raw, dict):
            raise ValueError("audition comparison must be a mapping")
        identifier = str(raw.get("id") or "")
        if not identifier or identifier in result:
            raise ValueError("audition comparison IDs must be unique")
        audio = raw.get("audio")
        if not isinstance(audio, dict):
            raise ValueError(f"comparison {identifier} has no audio mapping")
        if not all(isinstance(audio.get(key), str) for key in ("direct", "predicted")):
            raise ValueError(f"comparison {identifier} has incomplete audio")
        result[identifier] = raw
    return result


def _resolve_asset(manifest_path: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("audition audio path must be a non-empty string")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("audition audio path escapes its package")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(
            f"audition audio sample rate differs: {actual_rate}"
        )
    if audio.shape[1] != 1:
        raise ValueError(f"audition audio must be mono: {path}")
    mono = np.ascontiguousarray(audio[:, 0])
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"audition audio is empty or non-finite: {path}")
    return mono


def _render_gain(row: dict[str, Any], identifier: str) -> float:
    gain = float(row.get("shared_gain", 1.0))
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"comparison {identifier} has invalid shared gain")
    return gain


def _write_audio(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        subtype="PCM_16",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_blind_audition(
    baseline_manifest: str | Path,
    candidate_manifest: str | Path,
    output_root: str | Path,
    *,
    seed: int = 20260724,
) -> dict[str, object]:
    baseline_path = Path(baseline_manifest)
    candidate_path = Path(candidate_manifest)
    baseline = _load_manifest(baseline_path)
    candidate = _load_manifest(candidate_path)
    for field in ("sample_rate", "latent_hop", "future_frames"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"audition {field} contracts do not match")
    sample_rate = int(baseline["sample_rate"])

    baseline_rows = _comparison_map(baseline)
    candidate_rows = _comparison_map(candidate)
    identifiers = sorted(baseline_rows)
    if set(identifiers) != set(candidate_rows):
        raise ValueError("baseline and candidate comparison IDs do not match")

    destination = Path(output_root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"blind audition output is not empty: {destination}")
    audio_root = destination / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    shuffled = list(range(len(identifiers)))
    random.Random(seed).shuffle(shuffled)
    baseline_on_a = set(shuffled[::2])
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for index, identifier in enumerate(identifiers):
        base_row = baseline_rows[identifier]
        candidate_row = candidate_rows[identifier]
        for field in ("category", "split", "sample_count"):
            if base_row.get(field) != candidate_row.get(field):
                raise ValueError(
                    f"comparison {identifier} {field} does not match"
                )

        base_direct = _resolve_asset(
            baseline_path,
            base_row["audio"]["direct"],
        )
        candidate_direct = _resolve_asset(
            candidate_path,
            candidate_row["audio"]["direct"],
        )
        base_gain = _render_gain(base_row, identifier)
        candidate_gain = _render_gain(candidate_row, identifier)
        base_direct_audio = _read_audio(base_direct, sample_rate) / base_gain
        candidate_direct_audio = (
            _read_audio(candidate_direct, sample_rate) / candidate_gain
        )
        if (
            base_direct_audio.shape != candidate_direct_audio.shape
            or not np.allclose(
                base_direct_audio,
                candidate_direct_audio,
                rtol=2.0e-4,
                atol=1.0e-4,
            )
        ):
            raise ValueError(f"comparison {identifier} direct audio differs")
        direct_audio = 0.5 * (
            base_direct_audio + candidate_direct_audio
        )

        base_prediction = _resolve_asset(
            baseline_path,
            base_row["audio"]["predicted"],
        )
        candidate_prediction = _resolve_asset(
            candidate_path,
            candidate_row["audio"]["predicted"],
        )
        prediction_by_role = {
            "baseline": _read_audio(
                base_prediction,
                sample_rate,
            ) / base_gain,
            "candidate": _read_audio(
                candidate_prediction,
                sample_rate,
            ) / candidate_gain,
        }
        if any(
            prediction.shape != direct_audio.shape
            for prediction in prediction_by_role.values()
        ):
            raise ValueError(
                f"comparison {identifier} audio lengths do not match"
            )
        assignment = (
            {"A": "baseline", "B": "candidate"}
            if index in baseline_on_a
            else {"A": "candidate", "B": "baseline"}
        )
        peak = max(
            float(np.max(np.abs(direct_audio))),
            *(
                float(np.max(np.abs(prediction)))
                for prediction in prediction_by_role.values()
            ),
        )
        if not math.isfinite(peak) or peak <= 0.0:
            raise ValueError(f"comparison {identifier} audio is silent")
        shared_gain = 0.95 / peak
        public_id = f"trial-{index + 1:02d}"
        relative_audio = {
            "direct": f"audio/{public_id}-direct.wav",
            "A": f"audio/{public_id}-A.wav",
            "B": f"audio/{public_id}-B.wav",
        }
        _write_audio(
            destination / relative_audio["A"],
            prediction_by_role[assignment["A"]] * shared_gain,
            sample_rate,
        )
        _write_audio(
            destination / relative_audio["B"],
            prediction_by_role[assignment["B"]] * shared_gain,
            sample_rate,
        )
        _write_audio(
            destination / relative_audio["direct"],
            direct_audio * shared_gain,
            sample_rate,
        )
        public_rows.append(
            {
                "id": public_id,
                "category": base_row.get("category"),
                "split": base_row.get("split"),
                "sample_count": base_row.get("sample_count"),
                "shared_gain": shared_gain,
                "audio": relative_audio,
            }
        )
        private_rows.append(
            {
                "id": public_id,
                "source_id": identifier,
                "assignment": assignment,
            }
        )

    public_manifest: dict[str, object] = {
        "schema": 1,
        "title": "Z-RAVE blind rollout audition",
        "random_seed": seed,
        "sample_rate": baseline["sample_rate"],
        "latent_hop": baseline["latent_hop"],
        "future_frames": baseline["future_frames"],
        "rows": public_rows,
    }
    answer_key: dict[str, object] = {
        "schema": 1,
        "random_seed": seed,
        "models": {
            "baseline": baseline.get("transformer"),
            "candidate": candidate.get("transformer"),
        },
        "rows": private_rows,
    }
    _write_json(destination / "blind-manifest.json", public_manifest)
    _write_json(destination / "blind-answer-key.json", answer_key)
    (destination / "index.html").write_text(_PAGE, encoding="utf-8")
    return {
        "rows": public_rows,
        "public_manifest": public_manifest,
        "answer_key": answer_key,
        "output_root": str(destination.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic blind Z-RAVE A/B audition."
    )
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_blind_audition(
        args.baseline_manifest,
        args.candidate_manifest,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
