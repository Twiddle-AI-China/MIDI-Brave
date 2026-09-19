/**
 * Atlas Flow — monochrome instrument.
 *
 * Left: the atlas. Every cell is one preset's 128D timbre anchor projected by
 * the 8D PCA; the pointer is the coordinate. Nothing on the map moves except
 * the voice marker, which chases the pointer at the morph rate.
 *
 * Right: one of several analyser views of exactly what you are hearing.
 * Between them sits a three-band compressor, because preset loudness is one of
 * the model's failed quality gates and comparing timbres at wildly different
 * levels is pointless.
 */

const $ = selector => document.querySelector(selector);
const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = value => `${NOTES[value % 12]}${Math.floor(value / 12) - 1}`;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const KEYS = {a: 48, w: 49, s: 50, e: 51, d: 52, f: 53, t: 54, g: 55, y: 56, h: 57, u: 58, j: 59, k: 60};

// Not every browser decodes Ogg Vorbis (Safari notably), and an undecodable
// clip is silence with an error. Ask the browser what it can play.
const AUDIO_FORMAT = (() => {
  const probe = document.createElement('audio');
  return probe.canPlayType('audio/ogg; codecs=vorbis') ? 'ogg' : 'mp3';
})();
const pickAudio = payload =>
  (payload.urls && payload.urls[AUDIO_FORMAT]) || payload.url;
// Evaluation rows name their clips .ogg; the server transcodes on demand, so
// asking for the other extension is enough.
const auditionUrl = path => path.replace(/\.ogg$/, `.${AUDIO_FORMAT}`);

const INK = '#f2f2f2';
const CELL = '#d8d8d8';
const LONER = '#565656';

let status = null;
let evaluation = null;
let cells = [];
let coordinate = null;
let hovered = null;
let pointer = null;
let sounding = null;              // the cell the running voice actually landed on
const staticLayer = {canvas: document.createElement('canvas'), ready: false};
let selectedTest = null;
let takes = [];

const live = {
  socket: null, seq: 0, connected: false, lifecycle: '离线', planMs: 0,
  bytes: 0, since: 0, rate: 0, buffered: 0, underruns: 0, seenUnderruns: 0,
  timer: 0, audible: null, chain: null, fileChain: null, context: null,
  fileContext: null, player: null,
  // Voice state is shown through a slow gate: telemetry arrives every ~370 ms
  // and the raw label flickers between held_sustain and release as the pointer
  // moves, which reads as a fault rather than as information.
  shown: '离线', pending: '离线', gate: 0,
  // Telemetry arrives a few times a second; the marker is eased toward it every
  // animation frame so the voice glides instead of stepping.
  shownAudible: null, motion: 0, loop: null,
  offTimer: 0, inside: false, recorder: null, chunks: [],
};

/* ---------- map ---------- */

const map = $('#map');
const paint = map.getContext('2d');

function layout() {
  const ratio = window.devicePixelRatio || 1;
  const box = map.getBoundingClientRect();
  map.width = Math.round(box.width * ratio);
  map.height = Math.round(box.height * ratio);
  paint.setTransform(ratio, 0, 0, ratio, 0, 0);
  const pad = 64;
  const scaleX = (box.width - 2 * pad) / 2;
  const scaleY = (box.height - 2 * pad) / 2;
  map.view = {
    width: box.width, height: box.height,
    toPixel: value => [box.width / 2 + value[0] * scaleX, box.height / 2 - value[1] * scaleY],
    toNormal: (x, y) => [
      clamp((x - box.width / 2) / scaleX, -1, 1),
      clamp((box.height / 2 - y) / scaleY, -1, 1),
    ],
  };
  for (const cell of cells) [cell.x, cell.y] = map.view.toPixel(cell.pca);
  staticLayer.ready = false;
  drawMap();
}

/**
 * The colonies and the cells never move, so they are drawn once into an
 * offscreen canvas. Fifty radial gradients are far too expensive to repaint at
 * animation rate, and the voice marker needs animation rate to look like
 * motion rather than teleportation.
 */
function paintStatic() {
  const view = map.view;
  if (!view) return;
  const ratio = window.devicePixelRatio || 1;
  const layer = staticLayer.canvas;
  layer.width = Math.round(view.width * ratio);
  layer.height = Math.round(view.height * ratio);
  const ink = layer.getContext('2d');
  ink.setTransform(ratio, 0, 0, ratio, 0, 0);
  ink.clearRect(0, 0, view.width, view.height);

  // A sparse lattice so empty atlas regions read as space rather than void.
  ink.fillStyle = 'rgba(242, 242, 242, 0.055)';
  for (let x = view.width / 2 % 42; x < view.width; x += 42) {
    for (let y = view.height / 2 % 42; y < view.height; y += 42) {
      ink.fillRect(Math.round(x), Math.round(y), 1, 1);
    }
  }

  // Axis rails: the map is a projection with units, so label them.
  ink.font = '9px ui-monospace, monospace';
  ink.fillStyle = 'rgba(242, 242, 242, 0.30)';
  ink.strokeStyle = 'rgba(242, 242, 242, 0.14)';
  ink.lineWidth = 1;
  for (let step = -4; step <= 4; step++) {
    const value = step / 4;
    const [x, y] = view.toPixel([value, value]);
    ink.beginPath();                       // left rail, PC2
    ink.moveTo(0, Math.round(y) + 0.5);
    ink.lineTo(step % 2 ? 4 : 8, Math.round(y) + 0.5);
    ink.stroke();
    ink.textAlign = 'left';
    ink.textBaseline = 'middle';
    if (step % 2 === 0 && y < view.height - 70) ink.fillText(value.toFixed(1), 12, y);
    ink.beginPath();                       // bottom rail, PC1
    ink.moveTo(Math.round(x) + 0.5, view.height);
    ink.lineTo(Math.round(x) + 0.5, view.height - (step % 2 ? 4 : 8));
    ink.stroke();
    ink.textAlign = 'center';
    ink.textBaseline = 'bottom';
    // The bottom-left corner belongs to the [VOICE]/[ATLAS] readout; a tick
    // label there just collides with it.
    if (step % 2 === 0 && x > 300) ink.fillText(value.toFixed(1), x, view.height - 12);
  }
  ink.textBaseline = 'alphabetic';

  ink.globalCompositeOperation = 'lighter';
  for (const cell of cells) {
    const radius = cell.loner ? 34 : 66;
    const glow = ink.createRadialGradient(cell.x, cell.y, 0, cell.x, cell.y, radius);
    glow.addColorStop(0, `rgba(255, 255, 255, ${cell.loner ? 0.035 : 0.055})`);
    glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ink.fillStyle = glow;
    ink.beginPath();
    ink.arc(cell.x, cell.y, radius, 0, Math.PI * 2);
    ink.fill();
  }
  ink.globalCompositeOperation = 'source-over';
  staticLayer.ready = true;
}

function drawMap() {
  const view = map.view;
  if (!view) return;
  paint.clearRect(0, 0, view.width, view.height);
  if (!staticLayer.ready) paintStatic();
  paint.drawImage(staticLayer.canvas, 0, 0, view.width, view.height);


  // A double ring says "this preset is what you are hearing", which a plain
  // white dot among fifty white dots cannot.
  if (sounding) {
    for (const radius of [13, 17]) {
      paint.beginPath();
      paint.arc(sounding.x, sounding.y, radius, 0, Math.PI * 2);
      paint.strokeStyle = 'rgba(242, 242, 242, 0.75)';
      paint.lineWidth = 1;
      paint.stroke();
    }
  }

  for (const cell of cells) {
    if (cell.test) {
      paint.beginPath();
      paint.arc(cell.x, cell.y, 11, 0, Math.PI * 2);
      paint.strokeStyle = cell === selectedTest ? INK : 'rgba(242, 242, 242, 0.34)';
      paint.lineWidth = 1;
      paint.stroke();
    }
    paint.beginPath();
    paint.arc(cell.x, cell.y, cell === hovered ? 6 : cell.loner ? 3 : 4.2, 0, Math.PI * 2);
    paint.fillStyle = cell === hovered ? INK : cell.loner ? LONER : CELL;
    paint.fill();
  }

  const voice = live.shownAudible && view.toPixel(live.shownAudible);
  const cursor = pointer && view.toPixel(pointer);

  // The leash makes the lag explicit: the voice is still travelling to the
  // coordinate you are pointing at.
  if (voice && cursor && Math.hypot(voice[0] - cursor[0], voice[1] - cursor[1]) > 14) {
    paint.strokeStyle = 'rgba(242, 242, 242, 0.22)';
    paint.setLineDash([2, 3]);
    paint.lineWidth = 1;
    paint.beginPath();
    paint.moveTo(cursor[0], cursor[1]);
    paint.lineTo(voice[0], voice[1]);
    paint.stroke();
    paint.setLineDash([]);
  }
  if (voice) {
    const glow = paint.createRadialGradient(voice[0], voice[1], 0, voice[0], voice[1], 40);
    glow.addColorStop(0, 'rgba(255, 255, 255, 0.16)');
    glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
    paint.fillStyle = glow;
    paint.beginPath();
    paint.arc(voice[0], voice[1], 40, 0, Math.PI * 2);
    paint.fill();
    paint.strokeStyle = INK;
    paint.lineWidth = 1.5;
    paint.beginPath();
    paint.arc(voice[0], voice[1], 6, 0, Math.PI * 2);
    paint.moveTo(voice[0] - 11, voice[1]);
    paint.lineTo(voice[0] - 8, voice[1]);
    paint.moveTo(voice[0] + 8, voice[1]);
    paint.lineTo(voice[0] + 11, voice[1]);
    paint.moveTo(voice[0], voice[1] - 11);
    paint.lineTo(voice[0], voice[1] - 8);
    paint.moveTo(voice[0], voice[1] + 8);
    paint.lineTo(voice[0], voice[1] + 11);
    paint.stroke();
  }
  if (cursor) {
    paint.beginPath();
    paint.arc(cursor[0], cursor[1], 11, 0, Math.PI * 2);
    paint.strokeStyle = 'rgba(242, 242, 242, 0.5)';
    paint.lineWidth = 1;
    paint.stroke();
  }
  if (hovered) {
    paint.fillStyle = INK;
    paint.font = '600 10px ui-monospace, monospace';
    paint.textAlign = 'center';
    paint.fillText(hovered.label, hovered.x, hovered.y - 15);
  }

  if (trajectory.points.length > 1) drawTrajectory(view);
  // The inset is a luxury; on a narrow map it would cover the atlas it annotates.
  if (sounding && view.width > 620) drawInset(view, sounding);
}

function drawTrajectory(view) {
  paint.strokeStyle = trajectory.playing
    ? 'rgba(242, 242, 242, 0.45)'
    : 'rgba(242, 242, 242, 0.28)';
  paint.lineWidth = 1;
  paint.setLineDash(trajectory.drawing ? [] : [5, 4]);
  paint.beginPath();
  trajectory.points.forEach((value, index) => {
    const [x, y] = view.toPixel(value);
    index ? paint.lineTo(x, y) : paint.moveTo(x, y);
  });
  paint.stroke();
  paint.setLineDash([]);

  for (const end of [trajectory.points[0], trajectory.points.at(-1)]) {
    const [x, y] = view.toPixel(end);
    paint.strokeStyle = 'rgba(242, 242, 242, 0.5)';
    paint.strokeRect(x - 3.5, y - 3.5, 7, 7);
  }

  if (trajectory.playing) {
    const place = pointAt(trajectory.phase);
    if (place) {
      const [x, y] = view.toPixel(place);
      paint.beginPath();
      paint.arc(x, y, 4, 0, Math.PI * 2);
      paint.fillStyle = INK;
      paint.fill();
    }
  }
}

/**
 * The map can only show two of the eight atlas dimensions, so the sounding
 * preset gets a framed readout of all eight, tied back to its ring by a
 * hairline. The 62% of variance PC1 and PC2 carry is the part you can see;
 * this is the rest.
 */
function drawInset(view, cell) {
  const width = 132;
  const height = 92;
  const left = view.width - width - 18;
  const top = 18;

  const [ringX, ringY] = [cell.x, cell.y];
  const anchorX = left;
  const anchorY = top + height / 2;
  if (Math.hypot(ringX - anchorX, ringY - anchorY) > 60) {
    // Mostly orthogonal, like an annotation rule rather than a drawn line, and
    // faint enough that it does not compete with the cells it crosses.
    const elbow = anchorX - 26;
    paint.strokeStyle = 'rgba(242, 242, 242, 0.13)';
    paint.lineWidth = 1;
    paint.setLineDash([3, 4]);
    paint.beginPath();
    paint.moveTo(ringX + 19, ringY);
    paint.lineTo(elbow, ringY);
    paint.lineTo(elbow, anchorY);
    paint.lineTo(anchorX, anchorY);
    paint.stroke();
    paint.setLineDash([]);
  }

  paint.fillStyle = 'rgba(0, 0, 0, 0.72)';
  paint.fillRect(left, top, width, height);
  paint.strokeStyle = 'rgba(242, 242, 242, 0.28)';
  paint.lineWidth = 1;
  paint.strokeRect(left + 0.5, top + 0.5, width, height);

  paint.font = '9px ui-monospace, monospace';
  paint.textAlign = 'left';
  paint.fillStyle = 'rgba(242, 242, 242, 0.55)';
  paint.fillText(`[8D] ${cell.label}`, left + 7, top + 13);

  const rows = cell.pca.length;
  const usable = width - 34;
  for (let axis = 0; axis < rows; axis++) {
    const y = top + 24 + axis * 8;
    paint.fillStyle = 'rgba(242, 242, 242, 0.34)';
    paint.fillText(`${axis + 1}`, left + 7, y + 3);
    const middle = left + 20 + usable / 2;
    paint.fillStyle = 'rgba(242, 242, 242, 0.14)';
    paint.fillRect(left + 20, y, usable, 1);          // the zero line
    const extent = clamp(cell.pca[axis], -1, 1) * (usable / 2);
    paint.fillStyle = 'rgba(242, 242, 242, 0.85)';
    paint.fillRect(Math.min(middle, middle + extent), y - 2, Math.abs(extent), 4);
    paint.fillStyle = 'rgba(242, 242, 242, 0.30)';
    paint.fillRect(Math.round(middle), y - 4, 1, 8);  // centre tick
  }
}

/* ---------- audio chain: three-band compressor + analyser ---------- */

function buildChain(context) {
  const filter = (type, frequency) => {
    const node = context.createBiquadFilter();
    node.type = type;
    node.frequency.value = frequency;
    node.Q.value = Math.SQRT1_2;         // cascade two for Linkwitz-Riley 4
    return node;
  };
  const input = context.createGain();
  const drive = context.createGain();     // lifts the quiet model output into the
  const sum = context.createGain();       // compressor's working range
  const makeup = context.createGain();
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.45;

  const low = [filter('lowpass', 220), filter('lowpass', 220)];
  const mid = [filter('highpass', 220), filter('highpass', 220),
               filter('lowpass', 2600), filter('lowpass', 2600)];
  const high = [filter('highpass', 2600), filter('highpass', 2600)];
  const compressors = [context.createDynamicsCompressor(),
                       context.createDynamicsCompressor(),
                       context.createDynamicsCompressor()];

  input.connect(drive);
  const chainUp = (nodes, compressor) => {
    nodes.reduce((from, to) => (from.connect(to), to), drive).connect(compressor);
    compressor.connect(sum);
  };
  chainUp(low, compressors[0]);
  chainUp(mid, compressors[1]);
  chainUp(high, compressors[2]);
  // Drive plus makeup can easily exceed full scale, and a clipped demo tells you
  // nothing about the model, so the chain ends in a brick wall.
  const limiter = context.createDynamicsCompressor();
  limiter.threshold.value = -1.5;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.002;
  limiter.release.value = 0.12;
  sum.connect(makeup);
  makeup.connect(limiter);
  limiter.connect(analyser);
  analyser.connect(context.destination);

  return {context, input, drive, makeup, limiter, analyser, compressors, low, mid, high};
}

function applyCompressor(chain) {
  if (!chain) return;
  const on = $('#comp').getAttribute('aria-pressed') === 'true';
  const amount = Number($('#compAmount').value);
  const crossLow = Number($('#crossLow').value);
  const crossHigh = Number($('#crossHigh').value);
  const shape = [
    {attack: 0.020, release: 0.28},   // low
    {attack: 0.008, release: 0.16},   // mid
    {attack: 0.003, release: 0.09},   // high
  ];
  // The model renders quietly — a test take lands near -29 dBFS RMS, and each
  // band carries only a slice of that — so levelling needs gain in front of the
  // compressor, not merely a lower threshold. One slider drives both.
  chain.drive.gain.value = on ? 10 ** (20 * amount / 20) : 1;
  chain.compressors.forEach((compressor, index) => {
    compressor.knee.value = 12;
    compressor.attack.value = shape[index].attack;
    compressor.release.value = shape[index].release;
    compressor.threshold.value = on ? -28 : 0;
    compressor.ratio.value = on ? 2 + 6 * amount : 1;
  });
  for (const node of chain.low) node.frequency.value = crossLow;
  chain.mid[0].frequency.value = crossLow;
  chain.mid[1].frequency.value = crossLow;
  chain.mid[2].frequency.value = crossHigh;
  chain.mid[3].frequency.value = crossHigh;
  for (const node of chain.high) node.frequency.value = crossHigh;
  chain.makeup.gain.value = 10 ** (Number($('#makeup').value) / 20);
}

function activeChain() {
  return scope.source === '实时' ? live.chain : live.fileChain;
}

/* ---------- scope: several views of the same analyser ---------- */

const scope = {
  canvas: $('#scope'),
  strip: document.createElement('canvas'),
  width: 0, height: 0,
  analyser: null,
  source: '静音',
  running: false,
  view: 'spectrogram',
  freq: null, time: null, rows: null,
  last: 0, carry: 0,
  peaks: null,
  ridges: [],
  levels: [],
};

const mel = frequency => 2595 * Math.log10(1 + frequency / 700);
const melInverse = value => 700 * (10 ** (value / 2595) - 1);

function scopeLayout() {
  const box = scope.canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  scope.width = Math.max(140, Math.round(box.width));
  scope.height = Math.max(90, Math.round(box.height));
  scope.canvas.width = Math.round(box.width * ratio);
  scope.canvas.height = Math.round(box.height * ratio);
  scope.canvas.getContext('2d').setTransform(ratio, 0, 0, ratio, 0, 0);
  scope.strip.width = scope.width;
  scope.strip.height = scope.height;
  const strip = scope.strip.getContext('2d');
  strip.fillStyle = '#000';
  strip.fillRect(0, 0, scope.width, scope.height);
  scope.rows = null;
  scope.peaks = null;
}

function melRows(analyser, rate) {
  const bins = analyser.frequencyBinCount;
  const nyquist = rate / 2;
  const top = mel(nyquist);
  const bottom = mel(40);
  return Array.from({length: scope.height}, (_, row) => {
    const high = melInverse(bottom + (top - bottom) * (1 - row / scope.height));
    const low = melInverse(bottom + (top - bottom) * (1 - (row + 1) / scope.height));
    return [
      Math.max(0, Math.floor(low / nyquist * bins)),
      Math.max(1, Math.min(bins, Math.ceil(high / nyquist * bins))),
    ];
  });
}

function useAnalyser(analyser, label) {
  scope.analyser = analyser;
  scope.source = label;
  scope.rows = null;
  $('#scope-source').textContent = label;
  if (!scope.running) {
    scope.running = true;
    scope.last = performance.now();
    requestAnimationFrame(scopeFrame);
  }
}

function scopeFrame(now) {
  if (!scope.running) return;
  requestAnimationFrame(scopeFrame);
  const analyser = scope.analyser;
  if (!analyser) return;
  const elapsed = Math.min(0.2, (now - scope.last) / 1000);
  scope.last = now;
  const rate = analyser.context.sampleRate;
  if (!scope.freq || scope.freq.length !== analyser.frequencyBinCount) {
    scope.freq = new Uint8Array(analyser.frequencyBinCount);
    scope.time = new Uint8Array(analyser.fftSize);
    scope.rows = null;
  }
  analyser.getByteFrequencyData(scope.freq);
  analyser.getByteTimeDomainData(scope.time);
  if (!scope.rows) {
    scope.rows = melRows(analyser, rate);
  }
  $('#scope-range').textContent = ['wave', 'loudness'].includes(scope.view)
    ? '' : `40Hz–${(rate / 2000).toFixed(1)}kHz`;
  const context = scope.canvas.getContext('2d');
  context.fillStyle = '#000';
  context.fillRect(0, 0, scope.width, scope.height);
  VIEWS[scope.view](context, {rate, elapsed});
  updateCompressorMeter();
}

/** Spectrogram. Columns advance on wall-clock time, so a dropped animation
 *  frame stretches the picture instead of punching a black hole in it. */
function viewSpectrogram(context, {rate, elapsed}) {
  const strip = scope.strip.getContext('2d');
  scope.carry += elapsed * 64;
  const columns = clamp(Math.floor(scope.carry), 0, 10);
  scope.carry -= columns;
  if (columns > 0) {
    strip.globalCompositeOperation = 'copy';
    strip.drawImage(scope.strip, -columns, 0);
    strip.globalCompositeOperation = 'source-over';
    const dropped = live.underruns > live.seenUnderruns;
    live.seenUnderruns = live.underruns;
    for (let row = 0; row < scope.height; row++) {
      const [low, high] = scope.rows[row];
      let sum = 0;
      for (let bin = low; bin < high; bin++) sum += scope.freq[bin];
      const level = (sum / Math.max(1, high - low) / 255) ** 1.2;
      const shade = Math.round(clamp(level, 0, 1) * 255);
      strip.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      strip.fillRect(scope.width - columns, row, columns, 1);
    }
    if (dropped) {
      // Mark a real audio dropout instead of leaving an unexplained gap.
      strip.fillStyle = '#ffffff';
      strip.fillRect(scope.width - columns, scope.height - 3, Math.max(1, columns), 3);
    }
  }
  context.drawImage(scope.strip, 0, 0);
  frequencyGrid(context, rate);
  context.fillStyle = 'rgba(242, 242, 242, 0.5)';
  context.textAlign = 'right';
  context.font = '9px ui-monospace, monospace';
  context.fillText(`${(scope.width / 64).toFixed(1)} s`, scope.width - 6, scope.height - 6);
}

/** Instantaneous spectrum on a log axis, with a falling peak hold. */
function viewSpectrum(context, {rate, elapsed}) {
  const bins = scope.freq.length;
  const nyquist = rate / 2;
  const bars = Math.min(180, Math.floor(scope.width / 3));
  if (!scope.peaks || scope.peaks.length !== bars) scope.peaks = new Float32Array(bars);
  const lowest = 40;
  const span = Math.log2(nyquist / lowest);
  context.lineWidth = 1;
  for (let index = 0; index < bars; index++) {
    const from = lowest * 2 ** (span * index / bars);
    const to = lowest * 2 ** (span * (index + 1) / bars);
    const start = Math.max(0, Math.floor(from / nyquist * bins));
    const end = Math.max(start + 1, Math.min(bins, Math.ceil(to / nyquist * bins)));
    let peak = 0;
    for (let bin = start; bin < end; bin++) peak = Math.max(peak, scope.freq[bin]);
    const level = peak / 255;
    scope.peaks[index] = Math.max(level, scope.peaks[index] - elapsed * 0.55);
    const x = index / bars * scope.width;
    const width = scope.width / bars - 1;
    context.fillStyle = `rgba(242, 242, 242, ${0.25 + 0.65 * level})`;
    context.fillRect(x, scope.height * (1 - level), width, scope.height * level);
    context.fillStyle = 'rgba(242, 242, 242, 0.85)';
    context.fillRect(x, scope.height * (1 - scope.peaks[index]) - 1, width, 1);
  }
  logFrequencyGrid(context, rate, lowest);
}

/** Waterfall as stacked ridgelines — the shape of the timbre over time. */
function viewRidge(context, {rate, elapsed}) {
  const bands = 72;
  const bins = scope.freq.length;
  const nyquist = rate / 2;
  const lowest = 40;
  const span = Math.log2(nyquist / lowest);
  const line = new Float32Array(bands);
  for (let index = 0; index < bands; index++) {
    const from = lowest * 2 ** (span * index / bands);
    const to = lowest * 2 ** (span * (index + 1) / bands);
    const start = Math.max(0, Math.floor(from / nyquist * bins));
    const end = Math.max(start + 1, Math.min(bins, Math.ceil(to / nyquist * bins)));
    let peak = 0;
    for (let bin = start; bin < end; bin++) peak = Math.max(peak, scope.freq[bin]);
    line[index] = peak / 255;
  }
  scope.carry += elapsed * 22;
  while (scope.carry >= 1) {
    scope.carry -= 1;
    scope.ridges.unshift(line);
    if (scope.ridges.length > 46) scope.ridges.pop();
  }
  const step = scope.height / 52;
  const amplitude = scope.height / 4.6;
  context.save();
  context.beginPath();
  context.rect(0, 0, scope.width, scope.height - 12);
  context.clip();
  for (let index = scope.ridges.length - 1; index >= 0; index--) {
    const ridge = scope.ridges[index];
    const base = scope.height - 10 - index * step;
    const fade = 1 - index / scope.ridges.length;
    context.beginPath();
    context.moveTo(0, base);
    for (let band = 0; band < ridge.length; band++) {
      const x = band / (ridge.length - 1) * scope.width;
      context.lineTo(x, base - ridge[band] ** 1.3 * amplitude);
    }
    context.lineTo(scope.width, base);
    context.closePath();
    context.fillStyle = '#000';
    context.fill();
    context.strokeStyle = `rgba(242, 242, 242, ${0.10 + 0.75 * fade})`;
    context.lineWidth = 1;
    context.stroke();
  }
  context.restore();
  logFrequencyGrid(context, rate, lowest, true);
}

/** Triggered oscilloscope: lock on a rising zero crossing so the wave stands still. */
function viewWave(context) {
  const samples = scope.time;
  let trigger = 0;
  for (let index = 1; index < samples.length / 2; index++) {
    if (samples[index - 1] < 128 && samples[index] >= 128) { trigger = index; break; }
  }
  const count = Math.floor(samples.length / 2);
  context.strokeStyle = 'rgba(242, 242, 242, 0.14)';
  context.beginPath();
  context.moveTo(0, scope.height / 2);
  context.lineTo(scope.width, scope.height / 2);
  context.stroke();
  context.beginPath();
  for (let index = 0; index < count; index++) {
    const value = (samples[trigger + index] - 128) / 128;
    const x = index / (count - 1) * scope.width;
    const y = scope.height / 2 - value * scope.height * 0.42;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  }
  context.strokeStyle = INK;
  context.lineWidth = 1.2;
  context.stroke();
}

/** Loudness history with the compressor's gain reduction on top. */
function viewLoudness(context, {elapsed}) {
  let sum = 0;
  let peak = 0;
  for (const sample of scope.time) {
    const value = (sample - 128) / 128;
    sum += value * value;
    peak = Math.max(peak, Math.abs(value));
  }
  const rms = Math.sqrt(sum / scope.time.length);
  const chain = activeChain();
  const reduction = chain
    ? chain.compressors.reduce((total, item) => total + item.reduction, 0) / 3
    : 0;
  scope.carry += elapsed * 40;
  while (scope.carry >= 1) {
    scope.carry -= 1;
    scope.levels.push({
      rms: 20 * Math.log10(Math.max(rms, 1e-5)),
      peak: 20 * Math.log10(Math.max(peak, 1e-5)),
      reduction,
    });
    if (scope.levels.length > scope.width) scope.levels.shift();
  }
  const toY = db => scope.height * (1 - clamp((db + 72) / 72, 0, 1));
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'left';
  for (const db of [-6, -18, -30, -42, -54, -66]) {
    const y = toY(db);
    context.strokeStyle = 'rgba(242, 242, 242, 0.09)';
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(scope.width, y);
    context.stroke();
    context.fillStyle = 'rgba(242, 242, 242, 0.34)';
    context.fillText(`${db}`, 4, y - 3);
  }
  const trace = (key, style, width) => {
    context.beginPath();
    scope.levels.forEach((value, index) => {
      const x = scope.width - scope.levels.length + index;
      const y = toY(value[key]);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.strokeStyle = style;
    context.lineWidth = width;
    context.stroke();
  };
  trace('peak', 'rgba(242, 242, 242, 0.32)', 1);
  trace('rms', INK, 1.4);
  context.beginPath();
  scope.levels.forEach((value, index) => {
    const x = scope.width - scope.levels.length + index;
    const y = -value.reduction / 24 * scope.height;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.strokeStyle = 'rgba(242, 242, 242, 0.55)';
  context.setLineDash([3, 3]);
  context.lineWidth = 1;
  context.stroke();
  context.setLineDash([]);
  const latest = scope.levels.at(-1);
  if (latest) {
    context.textAlign = 'right';
    context.fillStyle = INK;
    context.fillText(`RMS ${latest.rms.toFixed(1)} dB   峰值 ${latest.peak.toFixed(1)} dB`,
      scope.width - 6, 14);
    context.fillStyle = 'rgba(242, 242, 242, 0.6)';
    context.fillText(`虚线＝压缩增益衰减 ${latest.reduction.toFixed(1)} dB（满幅 24 dB）`,
      scope.width - 6, 26);
  }
}

function frequencyGrid(context, rate) {
  const top = mel(rate / 2);
  const bottom = mel(40);
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'left';
  for (const frequency of [100, 250, 500, 1000, 2000, 4000, 8000, 16000]) {
    if (frequency >= rate / 2) continue;
    const y = scope.height * (1 - (mel(frequency) - bottom) / (top - bottom));
    context.strokeStyle = 'rgba(242, 242, 242, 0.10)';
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(scope.width, y);
    context.stroke();
    context.fillStyle = 'rgba(242, 242, 242, 0.36)';
    context.fillText(frequency >= 1000 ? `${frequency / 1000}k` : String(frequency), 4, y - 3);
  }
}

function logFrequencyGrid(context, rate, lowest, faint = false) {
  const span = Math.log2(rate / 2 / lowest);
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'center';
  for (const frequency of [100, 250, 500, 1000, 2000, 4000, 8000, 16000]) {
    if (frequency >= rate / 2) continue;
    const x = Math.log2(frequency / lowest) / span * scope.width;
    context.strokeStyle = `rgba(242, 242, 242, ${faint ? 0.05 : 0.09})`;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, scope.height - 12);
    context.stroke();
    context.fillStyle = 'rgba(242, 242, 242, 0.36)';
    context.fillText(frequency >= 1000 ? `${frequency / 1000}k` : String(frequency), x, scope.height - 3);
  }
}

const VIEWS = {
  spectrogram: viewSpectrogram,
  spectrum: viewSpectrum,
  ridge: viewRidge,
  wave: viewWave,
  loudness: viewLoudness,
};

function updateCompressorMeter() {
  const chain = activeChain();
  if (!chain) return;
  const reductions = chain.compressors.map(item => item.reduction);
  const limited = chain.limiter ? chain.limiter.reduction : 0;
  $('#compMeter').textContent =
    `GR ${reductions.map(value => value.toFixed(0).padStart(3)).join('/')}`
    + (limited < -0.5 ? ` 限幅${limited.toFixed(0)}` : '');
  $('#bands').textContent =
    `低 ${reductions[0].toFixed(1)} dB　中 ${reductions[1].toFixed(1)} dB　高 ${reductions[2].toFixed(1)} dB`;
}

/* ---------- control ---------- */

function currentControls() {
  return {
    pcaNormalized: coordinate.slice(),
    note: Number($('#note').value),
    velocity: Number($('#level').value),
    temperature: 0,
    morphSeconds: Number($('#morph').value),
  };
}

function nearestCellTo(place, radius = 26) {
  const [px, py] = map.view.toPixel(place);
  let best = null;
  let distance = Infinity;
  for (const cell of cells) {
    const measure = Math.hypot(cell.x - px, cell.y - py);
    if (measure < distance) { distance = measure; best = cell; }
  }
  return distance <= radius ? best : null;
}

const nearestCell = place => nearestCellTo(place, 26);

function moveTo(place, {snap = true} = {}) {
  pointer = place;
  const previous = hovered;
  hovered = snap ? nearestCell(place) : null;
  if (audioMode() === 'cached' && hovered && hovered !== previous && live.fileChain) {
    auditionCell(hovered);
  }
  coordinate = hovered ? hovered.pca.slice() : (() => {
    const value = coordinate.slice();
    value[0] = place[0];
    value[1] = place[1];
    return value;
  })();
  syncAxes();
  sendControl();
  updateReadout();
  drawMap();
}

/**
 * Rate-limit control messages — as a throttle, not a debounce.
 *
 * This was a debounce: every call reset the timer, so a continuous stream of
 * calls postponed the send forever. A real mouse drag emits faster than the
 * 55 ms window, and the trajectory walker emits every animation frame, so in
 * both cases nothing was ever sent and the voice sat still. A throttle sends
 * immediately when the window has passed and schedules the trailing edge
 * otherwise, which guarantees a message at least every 55 ms while moving.
 */
function sendControl() {
  if (audioMode() !== 'live' || !live.connected) return;
  const now = performance.now();
  const since = now - (live.lastSent || 0);
  if (since < 55) {
    if (!live.timer) {
      live.timer = setTimeout(() => { live.timer = 0; sendControl(); }, 55 - since);
    }
    return;
  }
  clearTimeout(live.timer);
  live.timer = 0;
  live.lastSent = now;
  (() => {
    if (live.socket?.readyState !== WebSocket.OPEN) return;
    // The runtime stops producing once the voice is idle, and while it is in
    // release it ignores control messages outright — so in either state a
    // control is silence you cannot explain. Only a start wakes it.
    if (live.lifecycle === 'idle' || live.lifecycle === 'release') {
      retrigger();
      return;
    }
    live.socket.send(JSON.stringify({type: 'control', seq: ++live.seq, ...currentControls()}));
  })();
}

function liveSend(message) {
  if (live.socket?.readyState === WebSocket.OPEN) live.socket.send(JSON.stringify(message));
}

function updateReadout() {
  const where = hovered
    ? `${hovered.label}   连通域 ${hovered.component}${hovered.loner ? '（单点）' : ''}`
    : '自由坐标';
  $('#readout').textContent =
    `[ATLAS] ${where}\n        PC1 ${coordinate[0].toFixed(2)}   PC2 ${coordinate[1].toFixed(2)}`;
}

function setWork(text) {
  $('#work').textContent = text || '';
}

/** Let a state label settle before showing it. */
function gateState(label) {
  live.pending = label;
  if (live.gate) return;
  live.gate = setTimeout(() => {
    live.gate = 0;
    if (live.shown !== live.pending) {
      live.shown = live.pending;
      updateStatus();
    }
  }, 450);
}

/** Ease the drawn voice position toward the last reported one, at 60 fps. */
function animateVoice(now) {
  live.motion = 0;
  const target = live.audible;
  const shown = live.shownAudible;
  if (!target) {
    if (shown) { live.shownAudible = null; drawMap(); }
    return;
  }
  if (!shown) {
    live.shownAudible = target.slice();
    drawMap();
    return;
  }
  const elapsed = Math.min(0.1, (now - (live.lastFrame || now)) / 1000);
  live.lastFrame = now;
  // Exponential approach with a ~90 ms time constant: fast enough to keep up
  // with a telemetry step, slow enough to read as movement.
  const factor = 1 - Math.exp(-elapsed / 0.09);
  let moved = 0;
  for (let axis = 0; axis < shown.length; axis++) {
    const delta = (target[axis] - shown[axis]) * factor;
    shown[axis] += delta;
    moved += Math.abs(delta);
  }
  // Keep the ring on the cell nearest what is *drawn*, so the two agree.
  const near = nearestCellTo(shown, Infinity);
  const changed = near !== sounding;
  if (changed) { sounding = near; updateVoice(); }
  if (moved > 1e-5 || changed) drawMap();
  if (moved > 1e-5) live.motion = requestAnimationFrame(animateVoice);
}

function nudgeVoice() {
  if (!live.motion) {
    live.lastFrame = performance.now();
    live.motion = requestAnimationFrame(animateVoice);
  }
}

function updateStatus() {
  const parts = [live.shown];
  if (live.connected) {
    parts.push(`规划 ${live.planMs.toFixed(0)}ms`);
    if (live.rate) parts.push(`${live.rate.toFixed(0)}kbit/s`);
    parts.push(`缓冲 ${live.buffered.toFixed(1)}s`);
    if (live.underruns) parts.push(`欠载 ${live.underruns}`);
    if (live.dropped) parts.push(`弃流 ${live.dropped}`);
  }
  $('#status').textContent = `[${live.connected ? 'LIVE' : 'IDLE'}] ` + parts.join('   ·   ');
}

function updateVoice() {
  // Cached mode has no socket, so this cannot be gated on the live connection:
  // what is sounding is whatever last started, from either path.
  if (!sounding || (!live.connected && audioMode() === 'live')) {
    $('#voice').textContent = '';
    return;
  }
  const resting = audioMode() === 'cached'
    ? !preview.source
    : ['idle', 'release', '离线'].includes(live.shown);
  $('#voice').textContent = sounding
    ? `[VOICE] ${sounding.label}${sounding.loner ? '（单点）' : ''}　${resting ? '刚才响的' : '正在响'}`
    : '';
}

/* ---------- live link ---------- */

function profile() {
  const list = status.streamProfiles || [];
  return list[Number($('#profile').value)] || list[0];
}

async function connect() {
  const wire = profile();
  const targetSeconds = Number($('#buffer').value);
  if (!live.context || live.context.sampleRate !== wire.sampleRate) {
    await live.context?.close();
    live.context = new AudioContext({sampleRate: wire.sampleRate, latencyHint: 'playback'});
    await live.context.audioWorklet.addModule('/live-player-worklet.js');
    live.player = new AudioWorkletNode(live.context, 'atlas-live-player', {outputChannelCount: [2]});
    live.chain = buildChain(live.context);
    live.player.connect(live.chain.input);
    live.player.port.onmessage = ({data}) => {
      if (data.type !== 'buffer') return;
      live.buffered = data.bufferedFrames / wire.sampleRate;
      live.underruns = data.underruns;
      live.dropped = data.dropped || 0;
      liveSend({type: 'buffer', bufferedFrames: data.bufferedFrames, underruns: data.underruns});
      updateStatus();
    };
  }
  applyCompressor(live.chain);
  await live.context.resume();
  live.player.port.postMessage({type: 'reset'});
  live.socket?.close();
  live.seq = 0;
  live.bytes = 0;
  live.since = performance.now();
  gateState('连接中'); live.shown = '连接中';
  updateStatus();
  useAnalyser(live.chain.analyser, '实时');

  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  live.socket = new WebSocket(`${scheme}//${location.host}/live/runtime`);
  live.socket.binaryType = 'arraybuffer';
  live.socket.onmessage = event => {
    if (event.data instanceof ArrayBuffer) {
      live.bytes += event.data.byteLength;
      const elapsed = performance.now() - live.since;
      if (elapsed > 2000) {
        live.rate = live.bytes * 8 / elapsed;
        live.bytes = 0;
        live.since = performance.now();
      }
      live.player.port.postMessage({type: 'pcm', buffer: event.data}, [event.data]);
      return;
    }
    const value = JSON.parse(event.data);
    if (value.type === 'ready') {
      live.connected = true;
      liveSend({
        type: 'start', seed: Number($('#seed').value) || 0,
        stream: {
          sampleRate: wire.sampleRate, channels: wire.channels,
          format: wire.format, targetSeconds,
        },
        ...currentControls(),
      });
    }
    if (value.type === 'stream') {
      live.player.port.postMessage({
        type: 'format', channels: value.channels, dtype: value.format,
        prime: Math.round(value.targetSeconds * value.sampleRate * 0.8),
        reprime: Math.round(0.3 * value.sampleRate),
        // Only a genuine runaway should be shed; ordinary overshoot is cheaper
        // to keep than to discard, because discarding is audible. Shedding also
        // hides the overflow from the producer — it reports the trimmed level,
        // so the producer keeps refilling — which is why this sits well clear
        // of the level the producer is aiming at.
        cap: Math.round(value.targetSeconds * value.sampleRate * 3.0),
      });
      $('#wire').textContent =
        `[WIRE] ${value.sampleRate / 1000}kHz ${value.channels === 1 ? 'mono' : 'stereo'} `
        + `${value.format} · 上限 ${value.kbitPerSecond} kbit/s`;
    }
    if (value.type === 'telemetry') {
      live.lifecycle = value.lifecycle;
      gateState(value.lifecycle);
      live.planMs = Number(value.planMs) || 0;
      live.audible = value.audiblePcaNormalized || live.audible;
      // Name the cell nearest what is *audible*, not the cell nearest the plan's
      // target: during a morph those are different points, and the ring has to
      // agree with the crosshair or it tells you the wrong preset.
      nudgeVoice();
      updateStatus();
    }
    if (value.type === 'error') {
      live.lifecycle = value.message;
      gateState(value.message);
      updateStatus();
    }
  };
  live.socket.onclose = event => {
    live.connected = false;
    live.audible = null;
    live.shownAudible = null;
    live.lifecycle = event.code === 1001 ? '已被另一个标签页接管' : '离线';
    // Closing the socket on purpose when switching to cached mode is not an
    // outage, so it must not report one.
    live.shown = audioMode() === 'cached' ? '预取模式' : live.lifecycle;
    sounding = null;
    updateStatus();
    updateVoice();
    drawMap();
  };
}

async function startAudio() {
  if (audioMode() === 'live') {
    await connect().catch(error => { live.shown = String(error.message || error); updateStatus(); });
    return;
  }
  await filePlayback().catch(() => null);
  await live.fileContext?.resume().catch(() => {});
  applyCompressor(live.fileChain);
  live.shown = '预取模式';
  updateStatus();
  warmPreviews();
}

function retrigger() {
  liveSend({type: 'start', seed: Number($('#seed').value) || 0, ...currentControls()});
  // Telemetry lags by up to ~370 ms; without this the next control tick would
  // still see 'idle' and start the voice a second time.
  live.lifecycle = 'planning';
}

/* ---------- rendered takes ---------- */

/** Render the coordinate you are on right now, at full rate, offline.
 *
 *  The live stream is decimated to survive the link, and the offline path uses
 *  the evaluation-grade 8-step solver, so this is the only way to hear the
 *  model at full bandwidth. It renders where you are — there is no path to
 *  draw and nothing to aim.
 */
async function renderCurrent() {
  const button = $('#render');
  button.disabled = true;
  setWork('渲染中…');
  if ($('#hold').getAttribute('aria-pressed') !== 'true') live.lifecycle = 'release';
  await primeAudio();          // while the click still counts as a gesture
  try {
    const response = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steps: [{
          pca: coordinate.map(value => Number(value.toFixed(4))),
          note: Number($('#note').value),
          seconds: 4.0,
        }],
        seed: Number($('#seed').value) || 0,
        velocity: Number($('#level').value),
        temperature: 0,
        morphSeconds: 1.0,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `渲染失败 ${response.status}`);
    addTake(payload, sounding ? sounding.label : '自由坐标');
    setWork(`${payload.seconds.toFixed(1)}s · ${(payload.bytes / 1024).toFixed(0)}KB`);
    play(pickAudio(payload), '渲染片段');
  } catch (error) {
    setWork(String(error.message || error));
  } finally {
    button.disabled = false;
  }
}

function addTake(payload, label) {
  takes = [{...payload, label}, ...takes.filter(item => item.id !== payload.id)].slice(0, 8);
  const host = $('#takes');
  host.textContent = '';
  for (const take of takes) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.textContent = `${take.label} ${take.seconds.toFixed(1)}s`;
    button.title = take.recorded
      ? `录制 · ${(take.bytes / 1024).toFixed(0)} KB`
      : `全速率渲染 · ${(take.bytes / 1024).toFixed(0)} KB · 峰值 ${take.peakDbfs} dBFS · GPU ${(take.renderMs / 1000).toFixed(1)} s`;
    button.addEventListener('click', () => {
      const url = take.recorded ? take.url : pickAudio(take);
      const label = take.recorded ? '录音' : take.label;
      if (looping() && !take.recorded) playLooping(url, label, take.take?.release ?? 2.4);
      else play(url, label);
    });
    host.append(button);
  }
}

/** Record what you are actually hearing, post-compressor. */
function toggleRecording() {
  const button = $('#rec');
  if (live.recorder) {
    live.recorder.stop();
    return;
  }
  if (!live.chain || typeof MediaRecorder === 'undefined') {
    setWork('这个浏览器不支持录制');
    return;
  }
  const destination = live.chain.context.createMediaStreamDestination();
  live.chain.limiter.connect(destination);
  live.chunks = [];
  const recorder = new MediaRecorder(destination.stream);
  recorder.ondataavailable = event => event.data.size && live.chunks.push(event.data);
  recorder.onstop = () => {
    live.chain.limiter.disconnect(destination);
    live.recorder = null;
    button.setAttribute('aria-pressed', 'false');
    button.textContent = '○ 录制';
    const blob = new Blob(live.chunks, {type: recorder.mimeType});
    const url = URL.createObjectURL(blob);
    const seconds = (performance.now() - started) / 1000;
    addTake({
      id: url, url, seconds, bytes: blob.size, recorded: true,
      peakDbfs: 0, renderMs: 0,
    }, '录音');
    setWork(`录了 ${seconds.toFixed(1)}s`);
  };
  const started = performance.now();
  recorder.start();
  live.recorder = recorder;
  button.setAttribute('aria-pressed', 'true');
  button.textContent = '● 录制中';
  setWork('录制中…');
}

async function filePlayback() {
  if (live.fileChain) return live.fileChain;
  const context = new AudioContext();
  const chain = buildChain(context);
  context.createMediaElementSource($('#player')).connect(chain.input);
  live.fileContext = context;
  live.fileChain = chain;
  return chain;
}

/**
 * Open the audio graph while a user gesture is still valid.
 *
 * Browsers grant audio on a *recent* gesture. Rendering a progression takes
 * 5-13 s, so creating the AudioContext after that fetch lands well outside the
 * grant: the context is born suspended, play() rejects, and — because that
 * rejection used to be swallowed — the result was silence with no error, for
 * that clip and every clip after it. Call this first, from the click itself.
 */
async function primeAudio() {
  const chain = await filePlayback().catch(error => {
    setWork(`播放链路失败 ${error.name || error}`);
    return null;
  });
  await live.fileContext?.resume().catch(() => {});
  return chain && live.fileContext?.state === 'running';
}

async function play(url, label = '渲染片段') {
  const player = $('#player');
  const chain = await filePlayback().catch(error => {
    setWork(`播放链路失败 ${error.name || error}`);
    return null;
  });
  applyCompressor(chain);
  await live.fileContext?.resume().catch(() => {});
  // The scope follows the element's own 'playing' event rather than this call
  // site: switching here races with whatever the element was doing before.
  live.playLabel = label;
  player.src = url;
  try {
    await player.play();
  } catch (error) {
    // NotAllowedError means the browser refused, not that the audio is broken.
    setWork(error.name === 'NotAllowedError'
      ? '浏览器拦截了播放，点一下页面再试'
      : `播放失败 ${error.name || error}`);
    return;
  }
  if (live.fileContext && live.fileContext.state !== 'running') {
    setWork('音频上下文被挂起，点一下页面恢复');
  }
}



/* ---------- cached previews: the responsive path ---------- */

/**
 * Hovering a cell should make a sound *now*. A live PCM stream cannot promise
 * that across this link: measured from the laptop, the median block arrives on
 * time but the worst gap is 3.9 s (22 s at a smaller buffer), while the same
 * probe run inside Kraken holds a 0.2 s buffer with zero underruns. The network
 * is the floor, not the model.
 *
 * So in cached mode nothing streams. Every preset is rendered once at full rate,
 * fetched, decoded into memory, and played locally on hover — zero latency, no
 * dropouts, and no traffic at all once warm. The cost is honest: it is discrete,
 * one clip per preset per pitch, with no continuous morph between them.
 */
const preview = {
  buffers: new Map(),     // "presetId:note" -> AudioBuffer
  pending: new Set(),
  source: null,
  gain: null,
  warming: false,
  note: null,
};

const previewKey = (cell, note) => `${cell.presetId}:${note}`;

function audioMode() {
  return $('#mode').value;
}

async function previewFor(cell, note) {
  const key = previewKey(cell, note);
  if (preview.buffers.has(key)) return preview.buffers.get(key);
  if (preview.pending.has(key)) return null;
  preview.pending.add(key);
  try {
    const chain = await filePlayback();
    const response = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steps: [{pca: cell.pca.map(v => Number(v.toFixed(4))), note, seconds: 2.5}],
        seed: Number($('#seed').value) || 0,
        velocity: 0.8, temperature: 0, morphSeconds: 0.5,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'render failed');
    const bytes = await (await fetch(pickAudio(payload))).arrayBuffer();
    const buffer = await chain.context.decodeAudioData(bytes);
    preview.buffers.set(key, buffer);
    return buffer;
  } catch (error) {
    setWork(`预取失败 ${error.message || error}`);
    return null;
  } finally {
    preview.pending.delete(key);
  }
}

function playPreview(buffer, cell = null) {
  const chain = live.fileChain;
  if (!chain || !buffer) return;
  stopPreview(0.012);
  const gain = chain.context.createGain();
  const source = chain.context.createBufferSource();
  source.buffer = buffer;
  source.connect(gain);
  gain.connect(chain.input);
  gain.gain.value = Number($('#level').value);
  source.onended = () => {
    if (preview.source === source) {
      preview.source = null;
      live.audible = null;
      live.shownAudible = null;
      updateVoice();
      drawMap();
    }
  };
  source.start();
  preview.source = source;
  preview.gain = gain;
  // Mark the voice only once it is actually running, or the HUD reads "just
  // played" for something that is playing right now.
  if (cell) {
    sounding = cell;
    live.audible = cell.pca;
    live.shownAudible = cell.pca.slice();   // a cached clip has no morph to trace
    updateVoice();
    drawMap();
  }
  useAnalyser(chain.analyser, '预取片段');
}

function stopPreview(fade = 0.02) {
  if (!preview.source) return;
  const {context} = live.fileChain;
  const when = context.currentTime;
  try {
    preview.gain.gain.cancelScheduledValues(when);
    preview.gain.gain.setValueAtTime(preview.gain.gain.value, when);
    preview.gain.gain.linearRampToValueAtTime(0.0001, when + fade);
    preview.source.stop(when + fade);
  } catch (_error) { /* already stopped */ }
  preview.source = null;
  preview.gain = null;
}

/** Warm every cell for the current pitch, a few at a time. */
async function warmPreviews() {
  const note = Number($('#note').value);
  if (preview.warming) return;
  preview.warming = true;
  preview.note = note;
  const queue = cells.filter(cell => !preview.buffers.has(previewKey(cell, note)));
  let done = cells.length - queue.length;
  const workers = Array.from({length: 3}, async () => {
    while (queue.length && preview.note === note) {
      const cell = queue.shift();
      await previewFor(cell, note);
      done += 1;
      setWork(`预取 ${done}/${cells.length}`);
    }
  });
  await Promise.all(workers);
  preview.warming = false;
  if (preview.note === note) {
    setWork(`预取就绪 ${cells.length} 个 · 悬停即响`);
    if (queue.length === 0) live.shown = '预取模式';
    updateStatus();
  } else {
    warmPreviews();
  }
}

async function auditionCell(cell) {
  const note = Number($('#note').value);
  const key = previewKey(cell, note);
  if (preview.buffers.has(key)) {
    playPreview(preview.buffers.get(key), cell);
    return;
  }
  setWork(`取 ${cell.label}…`);
  const buffer = await previewFor(cell, note);
  if (buffer && hovered === cell) playPreview(buffer, cell);
  if (buffer) setWork('');
}



/* ---------- drawn trajectories ---------- */

/**
 * A path the voice travels, rather than a path that renders something.
 *
 * You draw a stroke across the atlas; the voice then walks it end to end and
 * back, and the timbre interpolates only along that route. The engine already
 * morphs between coordinates, so this only has to decide *which* coordinate to
 * ask for at each moment — the whole feature is client-side.
 *
 * Only PC1 and PC2 come from the stroke; the other six dimensions stay wherever
 * the panel has them, because the map cannot express them.
 */
const trajectory = {
  points: [],        // normalised [pc1, pc2] along the stroke
  lengths: [],       // cumulative arc length, for even-speed travel
  total: 0,
  drawing: false,
  playing: false,
  phase: 0,          // 0..1 along the path
  forward: true,
  frame: 0,
  last: 0,
};

const drawArmed = () => $('#draw').getAttribute('aria-pressed') === 'true';

function measurePath() {
  trajectory.lengths = [0];
  let total = 0;
  for (let index = 1; index < trajectory.points.length; index++) {
    const [ax, ay] = trajectory.points[index - 1];
    const [bx, by] = trajectory.points[index];
    total += Math.hypot(bx - ax, by - ay);
    trajectory.lengths.push(total);
  }
  trajectory.total = total;
}

/** Position at arc-length fraction `phase`, so speed is even along the stroke. */
function pointAt(phase) {
  const points = trajectory.points;
  if (points.length < 2) return points[0] || null;
  const wanted = clamp(phase, 0, 1) * trajectory.total;
  let index = 1;
  while (index < trajectory.lengths.length - 1 && trajectory.lengths[index] < wanted) index++;
  const before = trajectory.lengths[index - 1];
  const span = Math.max(1e-9, trajectory.lengths[index] - before);
  const ratio = clamp((wanted - before) / span, 0, 1);
  const [ax, ay] = points[index - 1];
  const [bx, by] = points[index];
  return [ax + (bx - ax) * ratio, ay + (by - ay) * ratio];
}

function walkTrajectory(now) {
  if (!trajectory.playing) return;
  trajectory.frame = requestAnimationFrame(walkTrajectory);
  const lap = Math.max(0.5, Number($('#lap').value));
  const elapsed = Math.min(0.25, (now - (trajectory.last || now)) / 1000);
  trajectory.last = now;
  // There and back, so a stroke reads as a sweep rather than a jump cut at the
  // end of every lap.
  trajectory.phase += (trajectory.forward ? 1 : -1) * elapsed / lap;
  if (trajectory.phase >= 1) { trajectory.phase = 1; trajectory.forward = false; }
  if (trajectory.phase <= 0) { trajectory.phase = 0; trajectory.forward = true; }
  const place = pointAt(trajectory.phase);
  if (!place) return;
  coordinate[0] = place[0];
  coordinate[1] = place[1];
  hovered = null;
  syncAxes();
  sendControl();
  updateReadout();
  drawMap();
}

function startTrajectory() {
  if (trajectory.points.length < 2) return;
  measurePath();
  trajectory.playing = true;
  trajectory.last = performance.now();
  setWork(`[PATH] ${trajectory.points.length} 点 · ${Number($('#lap').value).toFixed(1)}s 单程`);
  if (audioMode() === 'live' && !live.connected) {
    connect().catch(() => {});
  } else if (['idle', 'release'].includes(live.lifecycle)) {
    retrigger();
  }
  cancelAnimationFrame(trajectory.frame);
  trajectory.frame = requestAnimationFrame(walkTrajectory);
}

function stopTrajectory(clear = false) {
  trajectory.playing = false;
  cancelAnimationFrame(trajectory.frame);
  trajectory.frame = 0;
  if (clear) {
    trajectory.points = [];
    trajectory.lengths = [];
    trajectory.total = 0;
    trajectory.phase = 0;
    trajectory.forward = true;
  }
  drawMap();
}

/* ---------- looping a progression ---------- */

/**
 * Play a rendered take as an endless loop.
 *
 * Repeating the file directly would not do: every take ends with the model's
 * 2.4 s release, so each cycle would decay into silence and then restart. The
 * loop therefore ends where the last chord ends, and the seam is crossfaded —
 * splicing a sustained pad back to its own start is otherwise a click.
 *
 * It runs on its own source rather than the shared one, so a loop keeps playing
 * while you roam the map over it.
 */
async function playLooping(url, label, releaseSeconds) {
  const chain = await filePlayback().catch(() => null);
  if (!chain) return;
  await live.fileContext?.resume().catch(() => {});
  applyCompressor(chain);
  stopLoop(0.02);

  const bytes = await (await fetch(url)).arrayBuffer();
  const decoded = await chain.context.decodeAudioData(bytes);
  const rate = decoded.sampleRate;
  const release = Math.max(0, releaseSeconds || 0);
  const crossfade = Math.min(0.25, decoded.duration * 0.05);
  const loopEnd = Math.max(crossfade + 0.1, decoded.duration - release);

  // Rebuild the buffer with its own tail mixed into its head, so the splice at
  // the seam is a crossfade instead of a step.
  const looped = chain.context.createBuffer(decoded.numberOfChannels,
    Math.floor(loopEnd * rate), rate);
  const span = Math.floor(crossfade * rate);
  for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
    const source = decoded.getChannelData(channel);
    const target = looped.getChannelData(channel);
    target.set(source.subarray(0, looped.length));
    for (let index = 0; index < span; index++) {
      const fade = index / span;
      const tail = source[Math.floor(loopEnd * rate) + index] || 0;
      target[index] = target[index] * fade + tail * (1 - fade);
    }
  }

  const gain = chain.context.createGain();
  gain.gain.value = Number($('#level').value);
  const source = chain.context.createBufferSource();
  source.buffer = looped;
  source.loop = true;
  source.loopStart = 0;
  source.loopEnd = looped.duration;
  source.connect(gain);
  gain.connect(chain.input);
  source.start();
  live.loop = {source, gain};
  useAnalyser(chain.analyser, `${label} ↻`);
  setWork(`${label} 循环中 · ${looped.duration.toFixed(1)}s/圈 · 停止或再按循环可结束`);
}

function stopLoop(fade = 0.05) {
  if (!live.loop) return;
  const {context} = live.fileChain;
  const when = context.currentTime;
  try {
    live.loop.gain.gain.cancelScheduledValues(when);
    live.loop.gain.gain.setValueAtTime(live.loop.gain.gain.value, when);
    live.loop.gain.gain.linearRampToValueAtTime(0.0001, when + fade);
    live.loop.source.stop(when + fade);
  } catch (_error) { /* already stopped */ }
  live.loop = null;
}

const looping = () => $('#loop').getAttribute('aria-pressed') === 'true';

/* ---------- chord progressions ---------- */

// Degrees in a major key, written the way they are spoken: 4361 is IV-iii-vi-I.
const PROGRESSIONS = [
  {name: '4361', degrees: [4, 3, 6, 1]},
  {name: '1645', degrees: [1, 6, 4, 5]},
  {name: '4536', degrees: [4, 5, 3, 6]},
  {name: '6451', degrees: [6, 4, 5, 1]},
  {name: '1564', degrees: [1, 5, 6, 4]},
  {name: '2516', degrees: [2, 5, 1, 6]},
  {name: '卡农 15634125', degrees: [1, 5, 6, 3, 4, 1, 2, 5]},
];

const MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11];
// Triad quality per degree of a major scale: I ii iii IV V vi vii°.
const THIRDS = [4, 3, 3, 4, 4, 3, 3];
const FIFTHS = [7, 7, 7, 7, 7, 7, 6];

/** Voice one degree as a triad inside the trained range (MIDI 36-71). */
function triad(root, degree) {
  const index = (degree - 1) % 7;
  let bottom = root + MAJOR_STEPS[index];
  while (bottom > 59) bottom -= 12;      // keep the chord in the pad's register
  while (bottom < 40) bottom += 12;
  const notes = [bottom, bottom + THIRDS[index], bottom + FIFTHS[index]];
  return notes.map(note => clamp(Math.round(note), 36, 71));
}

function buildProgressionPicker() {
  const select = $('#progression');
  PROGRESSIONS.forEach((item, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = item.name;
    select.append(option);
  });
  select.value = '0';
}

/** Render the selected progression at the timbre you are standing on.
 *
 *  The model is monophonic, so every chord is three voices rendered
 *  independently at the same coordinate and summed on the GPU side. That is a
 *  stack of monophonic pads, not a polyphonic instrument.
 */
async function playProgression() {
  const button = $('#play');
  const item = PROGRESSIONS[Number($('#progression').value)] || PROGRESSIONS[0];
  const root = Number($('#note').value);
  const seconds = clamp(20 / item.degrees.length, 1.5, 3.0);
  const chords = item.degrees.map(degree => triad(root, degree));
  // With a drawn path, the progression walks it: chord n sits at the point n
  // steps along the stroke, so the harmony and the timbre sweep together
  // instead of the chords all sharing one coordinate.
  const onPath = trajectory.points.length >= 2;
  if (onPath) measurePath();
  const placeFor = index => {
    if (!onPath) return coordinate;
    const at = pointAt(chords.length === 1 ? 0 : index / (chords.length - 1));
    const value = coordinate.slice();
    value[0] = at[0];
    value[1] = at[1];
    return value;
  };
  button.disabled = true;
  setWork(`${item.name} 渲染中…${onPath ? '（沿轨迹）' : ''}`);
  // A walking path is the live voice the user asked for; do not cut it off.
  if (!trajectory.playing && $('#hold').getAttribute('aria-pressed') !== 'true') {
    liveSend({type: 'note_off'});
    live.lifecycle = 'release';     // so the next roam starts the voice again
  }
  await primeAudio();          // while the click still counts as a gesture
  try {
    const response = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steps: chords.map((notes, index) => ({
          pca: placeFor(index).map(value => Number(value.toFixed(4))),
          notes,
          seconds: Number(seconds.toFixed(1)),
        })),
        seed: Number($('#seed').value) || 0,
        velocity: Number($('#level').value),
        temperature: 0,
        // Along a path the chords are travelling, so give the morph most of the
        // step to cross; standing still it only needs to settle.
        morphSeconds: onPath ? clamp(seconds * 0.8, 0.5, 5) : 0.5,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `渲染失败 ${response.status}`);
    addTake(payload, `${item.name} ${noteName(root)}${onPath ? ' ↗' : ''}`);
    setWork(`${item.name} · ${payload.voices} 声部 · ${payload.seconds.toFixed(1)}s`
      + `${onPath ? ' · 沿轨迹' : ''}`
      + `${payload.cached ? ' · 缓存' : ` · GPU ${(payload.renderMs / 1000).toFixed(1)}s`}`);
    $('#chordNote').textContent =
      `${item.name} 于 ${noteName(root)}：` + chords.map(c => c.map(noteName).join('-')).join('　');
    const release = payload.take?.release ?? 2.4;
    if (looping()) {
      await playLooping(pickAudio(payload), `进行 ${item.name}`, release);
    } else {
      play(pickAudio(payload), `进行 ${item.name}`);
    }
  } catch (error) {
    setWork(String(error.message || error));
  } finally {
    button.disabled = false;
  }
}

/** Silence everything: the live voice, and whatever clip is playing. */
function stopAll() {
  clearTimeout(live.offTimer);
  stopTrajectory();          // keeps the path so it can be resumed
  stopPreview(0.03);
  stopLoop(0.05);
  liveSend({type: 'stop'});
  const player = $('#player');
  player.pause();
  player.currentTime = 0;
  if (live.recorder) live.recorder.stop();
  live.lifecycle = 'idle';
  gateState('已停止');
  live.shown = '已停止';
  sounding = null;
  updateStatus();
  updateVoice();
  drawMap();
  setWork('');
}

/* ---------- held-out preset comparison ---------- */

function openTest(cell) {
  selectedTest = cell;
  const rows = (evaluation?.rows || []).filter(row => row.preset_id === cell.presetId);
  const host = $('#modes');
  host.hidden = false;
  let note = rows[0]?.note;
  let mode = 'source';

  const render = () => {
    host.textContent = '';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'chip';
    close.textContent = '×';
    close.addEventListener('click', () => {
      selectedTest = null;
      host.hidden = true;
      drawMap();
    });
    host.append(close);
    const label = document.createElement('span');
    label.className = 'mono muted';
    label.textContent = cell.label;
    host.append(label);
    for (const row of rows) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `chip${row.note === note ? ' on' : ''}`;
      chip.textContent = noteName(row.note);
      chip.addEventListener('click', () => { note = row.note; render(); swap(true); });
      host.append(chip);
    }
    for (const name of ['source', 'dynamic', 'static', 'flow']) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `chip${name === mode ? ' on' : ''}`;
      chip.textContent = name.toUpperCase();
      chip.addEventListener('click', () => { mode = name; render(); swap(false); });
      host.append(chip);
    }
  };

  const swap = restart => {
    const row = rows.find(item => item.note === note);
    if (!row) return;
    const player = $('#player');
    const position = restart ? 0 : player.currentTime;
    const playing = restart || !player.paused;
    liveSend({type: 'note_off'});
    play(auditionUrl(row.audio[mode]), `对照 ${mode.toUpperCase()}`).then(() => {
      player.currentTime = position;
      if (!playing) player.pause();
    });
  };

  render();
  swap(true);
  drawMap();
}

/* ---------- panel ---------- */

function syncAxes() {
  document.querySelectorAll('#axes input').forEach((input, index) => {
    input.value = coordinate[index];
    input.nextElementSibling.textContent = coordinate[index].toFixed(2);
  });
}

function buildPanel() {
  const select = $('#profile');
  (status.streamProfiles || []).forEach((wire, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${(wire.sampleRate / 1000).toFixed(wire.sampleRate % 1000 ? 2 : 0)}kHz `
      + `${wire.channels === 1 ? 'mono' : 'stereo'} ${wire.format} · ${(wire.kbitPerSecond / 1000).toFixed(2)} Mbit/s`;
    select.append(option);
  });
  // Measured on the Kraken link (Jagger -> Octopus -> Kraken, ~2.65 Mbit/s) in a
  // 30 s roam: 22.05 kHz mono int16 held 1.03x real time with no underruns;
  // 44.1 kHz mono managed 0.98x with two dropouts; stereo float32 saturates the
  // link at 0.96x. V100 planning is also slower than the retired GB10 host
  // (~150 ms re-plan vs ~70 ms), so default to headroom and trade up by hand.
  const preferred = (status.streamProfiles || []).findIndex(wire => wire.kbitPerSecond < 400);
  select.value = String(preferred < 0 ? 0 : preferred);

  const axes = $('#axes');
  (status.pcaAxes || []).forEach((axis, index) => {
    const share = (status.pcaExplained || [])[index];
    const label = document.createElement('label');
    label.innerHTML = `<span>${axis.label}${share ? ` ${(share * 100).toFixed(0)}%` : ''}</span>`
      + `<input type="range" min="-1" max="1" step="0.01" value="${coordinate[index]}">`
      + `<output>${coordinate[index].toFixed(2)}</output>`;
    const input = label.querySelector('input');
    input.addEventListener('input', () => {
      coordinate[index] = Number(input.value);
      label.querySelector('output').textContent = coordinate[index].toFixed(2);
      hovered = null;
      sendControl();
      updateReadout();
      drawMap();
    });
    axes.append(label);
  });
}

function showGates() {
  const gates = evaluation?.gates || {};
  const passed = Object.values(gates).filter(Boolean).length;
  const total = Object.keys(gates).length || 9;
  const badge = $('#gate');
  badge.textContent = `门禁 ${passed}/${total}`;
  badge.className = `chip${evaluation?.passed ? ' on' : ' bad'}`;
  $('#gates').textContent = Object.entries(gates)
    .map(([name, value]) => `${value ? '✓' : '✗'} ${name}`).join('　');
  $('#limits').textContent = (status.limitations || []).join(' ');
}

/* ---------- wiring ---------- */

/** ● when engaged, ○ when not — legible without colour. */
function markToggle(button) {
  const on = button.getAttribute('aria-pressed') === 'true';
  button.textContent = button.textContent.replace(/^[●○]/, on ? '●' : '○');
}

function wire() {
  const place = event => {
    const box = map.getBoundingClientRect();
    return map.view.toNormal(event.clientX - box.left, event.clientY - box.top);
  };

  map.addEventListener('pointermove', event => {
    const here = place(event);
    if (trajectory.drawing) {
      const last = trajectory.points.at(-1);
      // Thin the stroke: raw pointer samples are far denser than the path needs.
      if (!last || Math.hypot(here[0] - last[0], here[1] - last[1]) > 0.02) {
        trajectory.points.push(here);
        drawMap();
      }
      return;
    }
    if (trajectory.playing) return;      // the path owns the voice while it runs
    moveTo(here);
  });

  map.addEventListener('pointerdown', event => {
    map.setPointerCapture(event.pointerId);
    if (!live.connected && !live.fileChain) {
      $('#curtain').classList.add('gone');
      startAudio();
    }
    if (drawArmed()) {
      stopTrajectory(true);
      trajectory.drawing = true;
      trajectory.points = [place(event)];
      drawMap();
      return;
    }
    moveTo(place(event));
  });

  map.addEventListener('pointerup', event => {
    if (trajectory.drawing) {
      trajectory.drawing = false;
      if (trajectory.points.length >= 2) {
        startTrajectory();
      } else {
        setWork('轨迹太短，再画一条');
        trajectory.points = [];
      }
      drawMap();
      return;
    }
    if (trajectory.playing) return;
    moveTo(place(event));
    const cell = nearestCell(place(event));
    if (cell?.test) openTest(cell);
    else if (selectedTest) { selectedTest = null; $('#modes').hidden = true; drawMap(); }
  });

  // Leaving briefly should not cut the note: a release costs a fresh attack and
  // a fresh plan when the pointer comes straight back, which is what made the
  // state label thrash between held_sustain and release.
  map.addEventListener('pointerleave', () => {
    live.inside = false;
    pointer = null;
    hovered = null;
    drawMap();
    if (trajectory.playing || trajectory.drawing) return;
    if (audioMode() === 'cached') { stopPreview(0.08); return; }
    if (!live.connected || $('#hold').getAttribute('aria-pressed') === 'true') return;
    clearTimeout(live.offTimer);
    live.offTimer = setTimeout(() => {
      if (!live.inside) liveSend({type: 'note_off'});
    }, 700);
  });

  map.addEventListener('pointerenter', () => {
    live.inside = true;
    clearTimeout(live.offTimer);
    if (audioMode() !== 'live') return;
    if (live.connected && ['idle', 'release'].includes(live.lifecycle)) retrigger();
  });

  $('#note').addEventListener('input', event => {
    $('#noteOut').textContent = `${event.target.value} · ${noteName(Number(event.target.value))}`;
    sendControl();
    if (audioMode() === 'cached' && live.fileChain) warmPreviews();
  });
  $('#level').addEventListener('input', event => {
    $('#levelOut').textContent = Number(event.target.value).toFixed(2);
    sendControl();
  });
  $('#morph').addEventListener('input', event => {
    $('#morphOut').textContent = `${Number(event.target.value).toFixed(1)} s`;
    sendControl();
  });
  $('#buffer').addEventListener('input', event => {
    $('#bufferOut').textContent = `${Number(event.target.value).toFixed(1)} s`;
  });
  $('#hold').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    if (on && live.connected) liveSend({type: 'note_off'});
  });
  $('#comp').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    applyCompressor(live.chain);
    applyCompressor(live.fileChain);
  });
  for (const id of ['compAmount', 'crossLow', 'crossHigh', 'makeup']) {
    $(`#${id}`).addEventListener('input', event => {
      const value = Number(event.target.value);
      const out = $(`#${id}Out`);
      if (out) {
        out.textContent = id === 'compAmount' ? value.toFixed(2)
          : id === 'makeup' ? `${value.toFixed(1)} dB` : `${value} Hz`;
      }
      applyCompressor(live.chain);
      applyCompressor(live.fileChain);
    });
  }
  $('#view').addEventListener('change', event => {
    scope.view = event.target.value;
    scope.ridges = [];
    scope.levels = [];
    scope.carry = 0;
    scopeLayout();
  });
  $('#stop').addEventListener('click', stopAll);
  $('#play').addEventListener('click', playProgression);
  $('#draw').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
    markToggle(event.target);
    if (on) {
      stopTrajectory(true);        // disarming clears the path
      setWork('');
    } else {
      setWork('在图谱上拖一条线，松手后音色沿着它往返');
    }
  });
  $('#lap').addEventListener('input', event => {
    $('#lapOut').textContent = `${Number(event.target.value).toFixed(1)} s`;
    if (trajectory.playing) {
      setWork(`[PATH] ${trajectory.points.length} 点 · ${Number(event.target.value).toFixed(1)}s 单程`);
    }
  });
  $('#loop').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    markToggle(event.target);
    if (on) { stopLoop(); setWork(''); }
  });
  $('#rec').addEventListener('click', toggleRecording);
  $('#render').addEventListener('click', renderCurrent);
  $('#gear').addEventListener('click', () => {
    const panel = $('#panel');
    panel.hidden = !panel.hidden;
    $('#gear').classList.toggle('on', !panel.hidden);
  });
  $('#gate').addEventListener('click', () => $('#gear').click());
  $('#curtain').addEventListener('click', () => {
    $('#curtain').classList.add('gone');
    startAudio();
  });
  $('#mode').addEventListener('change', () => {
    stopPreview();
    if (audioMode() === 'live') {
      preview.note = null;
      connect().catch(error => { live.shown = String(error.message || error); updateStatus(); });
    } else {
      liveSend({type: 'stop'});
      live.socket?.close();
      live.shown = '预取模式';
      updateStatus();
      warmPreviews();
    }
  });
  const player = $('#player');
  player.addEventListener('play', () => {
    if ($('#hold').getAttribute('aria-pressed') !== 'true') liveSend({type: 'note_off'});
  });
  player.addEventListener('playing', () => {
    if (live.fileChain) useAnalyser(live.fileChain.analyser, live.playLabel || '渲染片段');
  });
  player.addEventListener('ended', () => {
    if (live.connected && live.chain) useAnalyser(live.chain.analyser, '实时');
    else $('#scope-source').textContent = '静音';
  });

  window.addEventListener('keydown', event => {
    const key = event.key.toLowerCase();
    if (event.repeat || event.target.matches('input, select') || !(key in KEYS)) return;
    $('#note').value = KEYS[key];
    $('#noteOut').textContent = `${KEYS[key]} · ${noteName(KEYS[key])}`;
    if (live.connected) retrigger();
  });
  window.addEventListener('resize', () => { layout(); scopeLayout(); });

  // Surface failures in the page. "my colleague saw some JS errors" is not
  // something anyone can act on; a line in the rail naming the error is.
  window.addEventListener('error', event => {
    setWork(`JS 错误 ${event.message || event.error}`);
  });
  window.addEventListener('unhandledrejection', event => {
    const reason = event.reason;
    setWork(`未处理错误 ${reason?.name || ''} ${reason?.message || reason}`.trim());
  });
}

/* ---------- boot ---------- */

async function boot() {
  status = await (await fetch('/api/status', {cache: 'no-store'})).json();
  evaluation = await (await fetch('/api/evaluation', {cache: 'no-store'})).json();
  const tests = new Set((evaluation.rows || []).map(row => row.preset_id));
  const sizes = new Map();
  for (const point of status.points) {
    sizes.set(point.component, (sizes.get(point.component) || 0) + 1);
  }
  cells = status.points.map(point => ({
    presetId: point.presetId,
    label: point.presetId.replace(/^serum_/, ''),
    pca: point.pcaNormalized,
    component: point.component,
    loner: sizes.get(point.component) === 1,
    test: tests.has(point.presetId),
    x: 0, y: 0,
  }));
  coordinate = (status.defaultPcaNormalized || Array(8).fill(0)).slice();
  buildPanel();
  buildProgressionPicker();
  showGates();
  wire();
  layout();
  scopeLayout();
  // The server knows whether it is on this machine or across a tunnel; take its
  // word for it rather than guessing from a hostname that is 127.0.0.1 either way.
  if (status.suggestedMode) {
    $('#mode').value = status.suggestedMode;
  }
  if (status.suggestedBufferSeconds) {
    $('#buffer').value = String(status.suggestedBufferSeconds);
    $('#bufferOut').textContent = `${status.suggestedBufferSeconds.toFixed(1)} s`;
  }
  $('#curtainNote').textContent = status.profile === 'local'
    ? `本机运行（${status.cuda}）：连续漫游，缓冲 ${status.suggestedBufferSeconds} s`
    : '预取模式：先把 50 个 preset 渲染好缓存，之后悬停零延迟';
  const explained = status.pcaExplained || [];
  if (explained.length) {
    $('#axis-note').textContent =
      `[PROJECTION] PC1 ${(explained[0] * 100).toFixed(0)}% × PC2 ${(explained[1] * 100).toFixed(0)}%`
      + ` ＝ 音色锚点方差的 ${((explained[0] + explained[1]) * 100).toFixed(0)}%`;
  }
  live.lifecycle = '离线';
  live.shown = `${status.cuda} · ${status.checkpoint}`;
  updateStatus();
}

// A small read-only window into the running instrument. Bug reports about this
// page have so far been "it is silent" or "it froze"; being able to ask for the
// actual numbers turns those into something answerable.
window.atlasDebug = () => ({
  mode: audioMode(),
  connected: live.connected,
  lifecycle: live.lifecycle,
  shownState: live.shown,
  target: coordinate && coordinate.slice(0, 2),
  audible: live.audible && live.audible.slice(0, 2),
  drawn: live.shownAudible && live.shownAudible.slice(0, 2),
  sounding: sounding && sounding.label,
  buffered: live.buffered,
  underruns: live.underruns,
  planMs: live.planMs,
  format: AUDIO_FORMAT,
  path: trajectory.points.length
    ? {points: trajectory.points.length, playing: trajectory.playing,
       phase: Number(trajectory.phase.toFixed(3))}
    : null,
});

boot().catch(error => {
  $('#status').textContent = `GPU 服务离线 · ${error.message || error}`;
});
