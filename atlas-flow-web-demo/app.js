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

const INK = '#f2f2f2';
const CELL = '#d8d8d8';
const LONER = '#565656';
const DRAG_THRESHOLD = 0.45;      // normalised map units before a drag is a route
const DRAG_MILLISECONDS = 250;

let status = null;
let evaluation = null;
let cells = [];
let coordinate = null;
let hovered = null;
let pointer = null;
let route = null;
let routeStarted = 0;
let shownRoute = null;            // only drawn while its take is playing
let selectedTest = null;
let takes = [];

const live = {
  socket: null, seq: 0, connected: false, lifecycle: '离线', planMs: 0,
  bytes: 0, since: 0, rate: 0, buffered: 0, underruns: 0, seenUnderruns: 0,
  timer: 0, audible: null, chain: null, fileChain: null, context: null,
  fileContext: null, player: null,
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
  drawMap();
}

function drawMap() {
  const view = map.view;
  if (!view) return;
  paint.clearRect(0, 0, view.width, view.height);

  // Colonies: cells of the same component bleed into one another. Nothing here
  // animates — it is a fixed picture of the atlas graph.
  paint.globalCompositeOperation = 'lighter';
  for (const cell of cells) {
    const radius = cell.loner ? 34 : 66;
    const glow = paint.createRadialGradient(cell.x, cell.y, 0, cell.x, cell.y, radius);
    glow.addColorStop(0, `rgba(255, 255, 255, ${cell.loner ? 0.035 : 0.055})`);
    glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
    paint.fillStyle = glow;
    paint.beginPath();
    paint.arc(cell.x, cell.y, radius, 0, Math.PI * 2);
    paint.fill();
  }
  paint.globalCompositeOperation = 'source-over';

  if (shownRoute) drawPath(shownRoute, 'rgba(242, 242, 242, 0.45)', [4, 4]);
  if (route) drawPath(route, 'rgba(242, 242, 242, 0.9)', []);

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

  const voice = live.audible && view.toPixel(live.audible);
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
    paint.beginPath();
    paint.arc(voice[0], voice[1], 5, 0, Math.PI * 2);
    paint.fillStyle = INK;
    paint.fill();
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
}

function drawPath(path, colour, dash) {
  if (path.length < 2) return;
  paint.strokeStyle = colour;
  paint.lineWidth = 1.5;
  paint.setLineDash(dash);
  paint.beginPath();
  path.forEach((value, index) => {
    const [x, y] = map.view.toPixel(value);
    index ? paint.lineTo(x, y) : paint.moveTo(x, y);
  });
  paint.stroke();
  paint.setLineDash([]);
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

function nearestCell(place) {
  const [px, py] = map.view.toPixel(place);
  let best = null;
  let distance = Infinity;
  for (const cell of cells) {
    const measure = Math.hypot(cell.x - px, cell.y - py);
    if (measure < distance) { distance = measure; best = cell; }
  }
  return distance <= 26 ? best : null;
}

function moveTo(place, {snap = true} = {}) {
  pointer = place;
  hovered = snap ? nearestCell(place) : null;
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

function sendControl() {
  if (!live.connected) return;
  clearTimeout(live.timer);
  live.timer = setTimeout(() => {
    if (live.socket?.readyState === WebSocket.OPEN) {
      live.socket.send(JSON.stringify({type: 'control', seq: ++live.seq, ...currentControls()}));
    }
  }, 55);
}

function liveSend(message) {
  if (live.socket?.readyState === WebSocket.OPEN) live.socket.send(JSON.stringify(message));
}

function updateReadout() {
  $('#readout').textContent = hovered
    ? `${hovered.label}   连通域 ${hovered.component}${hovered.loner ? '（单点）' : ''}\nPC1 ${coordinate[0].toFixed(2)}   PC2 ${coordinate[1].toFixed(2)}`
    : `自由坐标\nPC1 ${coordinate[0].toFixed(2)}   PC2 ${coordinate[1].toFixed(2)}`;
}

function updateStatus() {
  const parts = [live.lifecycle];
  if (live.connected) {
    parts.push(`规划 ${live.planMs.toFixed(0)}ms`);
    if (live.rate) parts.push(`${live.rate.toFixed(0)}kbit/s`);
    parts.push(`缓冲 ${live.buffered.toFixed(1)}s`);
    if (live.underruns) parts.push(`欠载 ${live.underruns}`);
  }
  $('#status').textContent = parts.join('   ·   ');
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
  live.lifecycle = '连接中';
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
      });
      $('#wire').textContent =
        `${value.sampleRate / 1000}kHz ${value.channels === 1 ? 'mono' : 'stereo'} `
        + `${value.format} · 上限 ${value.kbitPerSecond} kbit/s`;
    }
    if (value.type === 'telemetry') {
      live.lifecycle = value.lifecycle;
      live.planMs = Number(value.planMs) || 0;
      live.audible = value.audiblePcaNormalized || live.audible;
      updateStatus();
      drawMap();
    }
    if (value.type === 'error') {
      live.lifecycle = value.message;
      updateStatus();
    }
  };
  live.socket.onclose = event => {
    live.connected = false;
    live.audible = null;
    live.lifecycle = event.code === 1001 ? '已被另一个标签页接管' : '离线';
    updateStatus();
    drawMap();
  };
}

function retrigger() {
  liveSend({type: 'start', seed: Number($('#seed').value) || 0, ...currentControls()});
}

/* ---------- rendered takes ---------- */

function pathLength(path) {
  let total = 0;
  for (let index = 1; index < path.length; index++) {
    total += Math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]);
  }
  return total;
}

function sampleRoute(path) {
  const total = pathLength(path);
  const stops = Math.min(status.maxSteps || 4, Math.max(2, Math.round(total / 0.5) + 1));
  const wanted = Array.from({length: stops}, (_, index) => total * index / (stops - 1));
  const picked = [];
  let walked = 0;
  let cursor = 0;
  for (const distance of wanted) {
    while (cursor < path.length - 1) {
      const step = Math.hypot(path[cursor + 1][0] - path[cursor][0], path[cursor + 1][1] - path[cursor][1]);
      if (walked + step >= distance) break;
      walked += step;
      cursor++;
    }
    picked.push(path[Math.min(cursor, path.length - 1)]);
  }
  const seconds = clamp(10 / picked.length, 1.5, 3.5);
  return picked.map(place => {
    const value = coordinate.slice();
    value[0] = place[0];
    value[1] = place[1];
    return {
      pca: value.map(item => Number(item.toFixed(4))),
      note: Number($('#note').value),
      seconds: Number(seconds.toFixed(1)),
    };
  });
}

async function renderRoute(path) {
  const steps = sampleRoute(path);
  live.lifecycle = '离线渲染中…';
  updateStatus();
  if ($('#hold').getAttribute('aria-pressed') !== 'true') liveSend({type: 'note_off'});
  try {
    const response = await fetch('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steps, seed: Number($('#seed').value) || 0,
        velocity: Number($('#level').value), temperature: 0,
        morphSeconds: clamp(steps[0].seconds / 2, 0.5, 5),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `渲染失败 ${response.status}`);
    payload.path = path;
    addTake(payload, steps);
    play(payload.url, '渲染片段', path);
    live.lifecycle = live.connected ? 'held_sustain' : '离线';
  } catch (error) {
    live.lifecycle = String(error.message || error);
  }
  updateStatus();
}

function addTake(payload, steps) {
  takes = [{...payload, steps}, ...takes.filter(item => item.id !== payload.id)].slice(0, 8);
  const host = $('#takes');
  host.textContent = '';
  for (const take of takes) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.textContent = `${take.steps.length}点 ${take.seconds.toFixed(1)}s`;
    button.title = `${(take.bytes / 1024).toFixed(0)} KB · 峰值 ${take.peakDbfs} dBFS · GPU ${(take.renderMs / 1000).toFixed(1)} s`;
    button.addEventListener('click', () => play(take.url, '渲染片段', take.path));
    host.append(button);
  }
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

async function play(url, label = '渲染片段', path = null) {
  const player = $('#player');
  const chain = await filePlayback().catch(() => null);
  applyCompressor(chain);
  await live.fileContext?.resume().catch(() => {});
  shownRoute = path;
  drawMap();
  player.src = url;
  await player.play().catch(() => {});
  if (chain) useAnalyser(chain.analyser, label);
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
    play(row.audio[mode], `对照 ${mode.toUpperCase()}`).then(() => {
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

function wire() {
  const place = event => {
    const box = map.getBoundingClientRect();
    return map.view.toNormal(event.clientX - box.left, event.clientY - box.top);
  };

  map.addEventListener('pointermove', event => {
    const here = place(event);
    if (route) {
      const last = route.at(-1);
      if (Math.hypot(here[0] - last[0], here[1] - last[1]) > 0.04) route.push(here);
    }
    moveTo(here, {snap: !route});
  });

  map.addEventListener('pointerdown', event => {
    map.setPointerCapture(event.pointerId);
    if (!live.connected) {
      $('#curtain').classList.add('gone');
      connect().catch(error => { live.lifecycle = String(error.message || error); updateStatus(); });
      moveTo(place(event));
      return;
    }
    route = [place(event)];
    routeStarted = performance.now();
    drawMap();
  });

  map.addEventListener('pointerup', event => {
    const here = place(event);
    // A route has to be a deliberate stroke, otherwise every click would leave
    // a trajectory behind and trigger a render.
    const deliberate = route
      && pathLength(route) > DRAG_THRESHOLD
      && performance.now() - routeStarted > DRAG_MILLISECONDS;
    const drawn = deliberate ? route : null;
    route = null;
    if (drawn) {
      renderRoute(drawn);
      drawMap();
      return;
    }
    moveTo(here);
    const cell = nearestCell(here);
    if (cell?.test) openTest(cell);
    else if (selectedTest) { selectedTest = null; $('#modes').hidden = true; }
    if (live.connected) retrigger();
    drawMap();
  });

  map.addEventListener('pointerleave', () => {
    route = null;
    pointer = null;
    hovered = null;
    drawMap();
    if (live.connected && $('#hold').getAttribute('aria-pressed') !== 'true') {
      liveSend({type: 'note_off'});
    }
  });
  map.addEventListener('pointerenter', () => {
    if (live.connected && $('#hold').getAttribute('aria-pressed') !== 'true') retrigger();
  });

  $('#note').addEventListener('input', event => {
    $('#noteOut').textContent = `${event.target.value} · ${noteName(Number(event.target.value))}`;
    sendControl();
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
    if (on && live.connected) liveSend({type: 'note_off'});
  });
  $('#comp').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
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
  $('#gear').addEventListener('click', () => {
    const panel = $('#panel');
    panel.hidden = !panel.hidden;
    $('#gear').classList.toggle('on', !panel.hidden);
  });
  $('#gate').addEventListener('click', () => $('#gear').click());
  $('#curtain').addEventListener('click', () => {
    $('#curtain').classList.add('gone');
    connect().catch(error => { live.lifecycle = String(error.message || error); updateStatus(); });
  });
  const player = $('#player');
  player.addEventListener('play', () => {
    if ($('#hold').getAttribute('aria-pressed') !== 'true') liveSend({type: 'note_off'});
  });
  player.addEventListener('ended', () => {
    shownRoute = null;
    drawMap();
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
  showGates();
  wire();
  layout();
  scopeLayout();
  const explained = status.pcaExplained || [];
  if (explained.length) {
    $('#axis-note').textContent =
      `PC1 ${(explained[0] * 100).toFixed(0)}% × PC2 ${(explained[1] * 100).toFixed(0)}%`
      + ` ＝ 音色锚点方差的 ${((explained[0] + explained[1]) * 100).toFixed(0)}%`;
  }
  live.lifecycle = `${status.cuda} · ${status.checkpoint}`;
  updateStatus();
}

boot().catch(error => {
  $('#status').textContent = `GPU 服务离线 · ${error.message || error}`;
});
