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

/**
 * Canvas colours, read from the stylesheet rather than written twice.
 *
 * The map and the scopes are canvases, so CSS custom properties cannot reach
 * them; every shade used to be a literal rgba(242, 242, 242, x) spread across
 * thirty call sites. These are refreshed from the active theme's tokens by
 * applyTheme(), and `tint(alpha)` is what the drawing code asks for.
 */
let INK = '#f2f2f2';
let CELL = '#d8d8d8';
let LONER = '#565656';
let HOT = '#ffffff';
let INK_RGB = '242, 242, 242';
let BG_RGB = '0, 0, 0';
let BG = '#000000';

let INK_TRIPLE = [242, 242, 242];
let BG_TRIPLE = [0, 0, 0];

/**
 * The roles a theme colours, beyond plain ink.
 *
 * Structure -- the grid, the axis rails, the labels, the panel frames -- stays
 * neutral and uses tint(). These are the things that mean something: which
 * preset you are hearing, where you are pointing, the path you drew, which
 * cells the model never saw. Giving them one shared ink made the map readable
 * but flat, and made every "theme" a different shade of the same monochrome.
 */
const ROLES = ['cell', 'loner', 'edge', 'accent', 'accent2', 'path', 'test', 'warn'];
const PALETTE = {};
const TRIPLES = {};
let RAMP = [[0, 0, 0], [255, 255, 255]];

// Named tint, not ink: paintStatic and the scope strips both call their
// 2D context `ink`, which shadowed this in exactly the functions that draw.
const tint = alpha => `rgba(${INK_RGB}, ${alpha})`;
const shade = alpha => `rgba(${BG_RGB}, ${alpha})`;
/** A role colour at partial opacity, for the same role at lower emphasis. */
const role = (name, alpha = 1) => alpha >= 1
  ? PALETTE[name]
  : `rgba(${(TRIPLES[name] || INK_TRIPLE).join(', ')}, ${alpha})`;

function parseColor(value) {
  const text = String(value || '').trim();
  const hex = text.replace('#', '');
  if (/^[0-9a-f]{6}$/i.test(hex)) {
    return [0, 2, 4].map(at => parseInt(hex.slice(at, at + 2), 16));
  }
  const numbers = text.match(/[\d.]+/g);
  return numbers && numbers.length >= 3 ? numbers.slice(0, 3).map(Number) : [242, 242, 242];
}

/**
 * An opaque colour `level` of the way along the theme's ramp.
 *
 * The spectrogram scrolls its strip sideways and paints one fresh column per
 * frame, so it needs opaque pixels: a translucent fill would composite over
 * whatever scrolled underneath and smear the history. A ramp of several stops
 * rather than background-to-ink is what makes a spectrogram legible -- the
 * scientific colormaps exist because a single hue wastes most of its dynamic
 * range on shades the eye cannot separate.
 */
function ramp(level) {
  const t = clamp(level, 0, 1) * (RAMP.length - 1);
  const low = RAMP[Math.floor(t)];
  const high = RAMP[Math.min(RAMP.length - 1, Math.floor(t) + 1)];
  const f = t - Math.floor(t);
  const mix = channel => Math.round(low[channel] + (high[channel] - low[channel]) * f);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

// Set at import time, before the first paint and before wire() reads it to
// position the dropdown: doing this inside boot() meant the page came up in the
// stored theme while the settings menu still claimed monochrome.
document.documentElement.dataset.theme = localStorage.getItem('atlas.theme') || 'mono';

/** Themes live in the stylesheet; this copies the active one onto the canvas. */
function applyTheme(name) {
  const root = document.documentElement;
  if (name) {
    root.dataset.theme = name;
    localStorage.setItem('atlas.theme', name);
  }
  const style = getComputedStyle(root);
  const token = (key, fallback) => (style.getPropertyValue(key) || fallback).trim();
  INK_RGB = token('--ink-rgb', '242, 242, 242');
  BG_RGB = token('--bg-rgb', '0, 0, 0');
  INK_TRIPLE = parseColor(INK_RGB);
  BG_TRIPLE = parseColor(BG_RGB);
  INK = token('--ink', '#f2f2f2');
  BG = token('--bg', '#000000');
  for (const key of ROLES) {
    PALETTE[key] = token(`--${key}`, INK);
    TRIPLES[key] = parseColor(PALETTE[key]);
  }
  CELL = PALETTE.cell;
  LONER = PALETTE.loner;
  HOT = PALETTE.warn;
  RAMP = token('--ramp', '0 0 0 / 255 255 255').split('/').map(parseColor);
  // The map's background layer and every scope strip were painted in the old
  // palette, so both have to be thrown away rather than drawn over.
  staticLayer.ready = false;
  scopeLayout();
  drawMap();
}

let status = null;
let evaluation = null;
let cells = [];
let coordinate = null;
let hovered = null;
let pointer = null;
let sounding = null;              // the cell the running voice actually landed on
const staticLayer = {canvas: document.createElement('canvas'), ready: false};
let takes = [];

const live = {
  socket: null, seq: 0, connected: false, lifecycle: 'offline', planMs: 0,
  bytes: 0, since: 0, rate: 0, buffered: 0, underruns: 0, seenUnderruns: 0,
  timer: 0, audible: null, chain: null, fileChain: null, context: null,
  fileContext: null, player: null,
  // Voice state is shown through a slow gate: telemetry arrives every ~370 ms
  // and the raw label flickers between held_sustain and release as the pointer
  // moves, which reads as a fault rather than as information.
  shown: 'offline', pending: 'offline', gate: 0,
  // Telemetry arrives a few times a second; the marker is eased toward it every
  // animation frame so the voice glides instead of stepping.
  shownAudible: null, motion: 0, loop: null,
  offTimer: 0, inside: false, recorder: null, chunks: [],
};

/* ---------- language ---------- */

/**
 * Two languages, one string table.
 *
 * Static page text carries its English in a `data-en` attribute and its
 * Chinese in the markup, so the DOM is its own dictionary and only the strings
 * the script builds at runtime need a key here. Nothing in this table is ever
 * compared against: state is ASCII keys everywhere, and this turns keys into
 * words only at the moment of drawing them.
 */
const STRINGS = {
  'scope.silent':       ['静音', 'silent'],
  'scope.live':         ['实时', 'live'],
  'scope.clip':         ['渲染片段', 'clip'],
  'scope.preview':      ['预取片段', 'preview'],
  'scope.loop':         ['循环', 'loop'],
  'state.offline':      ['离线', 'offline'],
  'state.connecting':   ['连接中', 'connecting'],
  'state.cached':       ['预取模式', 'cached'],
  'state.stopped':      ['已停止', 'stopped'],
  'state.disconnected': ['未连接', 'not connected'],
  'state.taken_over':   ['已被另一个标签页接管', 'taken over by another tab'],
  'meta.plan':          ['规划', 'plan'],
  'meta.buffer':        ['缓冲', 'buffer'],
  'meta.underrun':      ['欠载', 'underrun'],
  'meta.dropped':       ['弃流', 'dropped'],
  'voice.loner':        ['（单点）', ' (isolated)'],
  'voice.sounding':     ['正在响', 'sounding'],
  'voice.resting':      ['刚才响的', 'last sounded'],
  'voice.component':    ['连通域', 'component'],
  'view.spectrogram':   ['频谱图 SPECTROGRAM', 'SPECTROGRAM'],
  'view.ridge':         ['瀑布 WATERFALL', 'WATERFALL'],
  'view.loudness':      ['响度 LOUDNESS', 'LOUDNESS'],
  'view.spectrum':      ['频谱 SPECTRUM', 'SPECTRUM'],
  'view.wave':          ['波形 OSCILLOSCOPE', 'OSCILLOSCOPE'],
  'scope.stream':       ['流', 'stream'],
  'level.peak':         ['峰值', 'peak'],
  'level.squeeze':      ['压缩', 'comp'],
  'level.limit':        ['限幅', 'limit'],
  'band.low':           ['低', 'low'],
  'band.mid':           ['中', 'mid'],
  'band.high':          ['高', 'high'],
  'wire.ceiling':       ['上限', 'ceiling'],
  'work.rendering':     ['渲染中…', 'rendering...'],
  'work.renderFailed':  ['渲染失败', 'render failed'],
  'work.freeCoord':     ['自由坐标', 'free coordinate'],
  'work.noRecorder':    ['这个浏览器不支持录制', 'this browser cannot record'],
  'work.recording':     ['录制中…', 'recording...'],
  'work.recorded':      ['录了', 'recorded'],
  'work.recordChip':    ['○ 录制', '○ REC'],
  'work.recordingChip': ['● 录制中', '● RECORDING'],
  'work.recordTake':    ['录音', 'recording'],
  'work.chainFailed':   ['播放链路失败', 'audio chain failed'],
  'work.blocked':       ['浏览器拦截了播放，点一下页面再试', 'the browser blocked playback, click the page and retry'],
  'work.suspended':     ['音频上下文被挂起，点一下页面恢复', 'audio context suspended, click the page to resume'],
  'work.playFailed':    ['播放失败', 'playback failed'],
  'work.prefetchFail':  ['预取失败', 'prefetch failed'],
  'work.prefetch':      ['预取', 'prefetching'],
  'work.prefetchDone':  ['预取就绪', 'prefetched'],
  'work.prefetchHint':  ['个 · 悬停即响', ' presets · hover to hear'],
  'work.fetching':      ['取', 'fetching'],
  'work.jsError':       ['JS 错误', 'JS error'],
  'work.unhandled':     ['未处理错误', 'unhandled error'],
  'take.recorded':      ['录制', 'recording'],
  'take.rendered':      ['全速率渲染', 'full-rate render'],
  'path.points':        ['点', 'points'],
  'path.oneWay':        ['s 单程', 's one way'],
  'path.hint':          ['在图谱上拖一条线，松手后音色沿着它往返',
                         'drag a line across the atlas; the timbre walks it and back'],
  'loop.running':       ['循环中', 'looping'],
  'loop.perLap':        ['s/圈 · 停止或再按循环可结束', 's/lap · stop, or press loop again, to end'],
  'prog.voices':        ['声部', 'voices'],
  'prog.cached':        ['缓存', 'cached'],
  'prog.onPath':        ['沿轨迹', 'along the path'],
  'prog.at':            ['于', 'in'],
  'prog.canon':         ['卡农 15634125', 'Canon 15634125'],
  'prog.label':         ['进行', 'progression'],
  'curtain.local':      ['本机运行', 'running locally'],
  'curtain.localTail':  ['：连续漫游，缓冲', ': continuous roaming, buffer'],
  'curtain.remote':     ['预取模式：先把 50 个 preset 渲染好缓存，之后悬停零延迟',
                         'prefetch mode: all 50 presets are rendered up front, then hovering is instant'],
  'gates.auto':         ['自动门禁', 'automatic gates'],
  'prog.progName':      ['进行', 'progression'],
  'work.pathShort':     ['轨迹太短，再画一条', 'that path is too short, draw another'],
  'drawer.open':        ['展开频谱', 'open the scope'],
  'drawer.shut':        ['收起频谱', 'collapse the scope'],
  'boot.failed':        ['启动失败', 'failed to start'],
};

const languages = ['zh', 'en'];
let language = localStorage.getItem('atlas.lang') === 'en' ? 'en' : 'zh';

const t = key => (STRINGS[key] || [key, key])[language === 'en' ? 1 : 0];

/** Most progressions are digits and need no translation; one is not. */
const progName = item => (item.key ? t(item.key) : item.name);

/**
 * Swap every translatable string on the page.
 *
 * The Chinese in the markup is the original; the English lives beside it in
 * `data-en` (and `data-info-en` for the hover explanations). Each element's
 * Chinese is captured once on first use, so this can be toggled any number of
 * times without the two languages contaminating each other.
 */
function applyLanguage() {
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
  localStorage.setItem('atlas.lang', language);
  for (const node of document.querySelectorAll('[data-en]')) {
    // A <label> wraps the control it names, so writing textContent would
    // delete the slider or select inside it. Only the element's own leading
    // text may be swapped.
    const own = [...node.childNodes]
      .find(item => item.nodeType === Node.TEXT_NODE && item.textContent.trim());
    const target = node.children.length && own ? own : node;
    if (node.dataset.zh === undefined) node.dataset.zh = target.textContent;
    target.textContent = language === 'en' ? node.dataset.en : node.dataset.zh;
  }
  for (const node of document.querySelectorAll('[data-info-en]')) {
    if (node.dataset.infoZh === undefined) node.dataset.infoZh = node.dataset.info;
    node.dataset.info = language === 'en' ? node.dataset.infoEn : node.dataset.infoZh;
  }
  // Anything the script drew itself has to be drawn again.
  if (scope.views.length) {
    // buildSlots hands back fresh views with no dimensions, and the frame loop
    // skips a view of width 0 -- so rebuilding without re-measuring is how the
    // visualiser went blank until the drawer was toggled.
    buildSlots(true);
    scopeLayout();
  }
  if (cells.length) {
    // The picker holds one translated name among the digits, so it has to be
    // relabelled too -- keeping the selection, which is the whole point.
    const chosen = $('#progression').value;
    buildProgressionPicker();
    if (chosen) $('#progression').value = chosen;
  }
  renderScopeSource();
  showGates();
  syncAxes();
  updateStatus();
  updateVoice();
  drawMap();
}

/* ---------- map ---------- */

/**
 * A k-nearest-neighbour graph over the atlas, computed in the full 8-D space
 * rather than in the two dimensions you can see.
 *
 * This is the honest version of what the fifty dots used to say. PC1 and PC2
 * carry 42.3% and 19.8% of the variance, so more than a third of what
 * separates two presets happens on axes this projection cannot draw. Two dots
 * sitting next to each other on screen are not necessarily neighbours, and the
 * graph is where that shows: an edge is dashed when most of the pair's real
 * distance lives in the six hidden axes, so its endpoints look far closer than
 * they are.
 */
const KNN = 3;
// A kNN edge on this atlas keeps a median of 0.36 of its true 8-D length once
// projected, so "loses something" describes almost every edge and is not worth
// drawing. Dashing the worst quarter (measured: 27% fall below this) makes the
// dash mean something you can act on.
const FOLDED = 0.25;
let edges = [];

const distance = (a, b, dims) => {
  let sum = 0;
  for (let i = 0; i < dims; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
};

function buildNeighbourGraph() {
  const seen = new Set();
  const found = [];
  cells.forEach((cell, index) => {
    const ranked = cells
      .map((other, j) => ({other, j, d: distance(cell.pca, other.pca, 8)}))
      .filter(item => item.j !== index)
      .sort((left, right) => left.d - right.d)
      .slice(0, KNN);
    cell.neighbours = ranked.map(item => item.other);
    for (const item of ranked) {
      const key = index < item.j ? `${index}:${item.j}` : `${item.j}:${index}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const flat = distance(cell.pca, item.other.pca, 2);
      found.push({
        a: cell, b: item.other, d: item.d,
        // How much of the separation survives the projection. Low means the
        // pair is drawn far closer together than it really is.
        shown: item.d > 1e-6 ? flat / item.d : 1,
      });
    }
  });
  const near = Math.min(...found.map(edge => edge.d));
  const far = Math.max(...found.map(edge => edge.d));
  const span = Math.max(far - near, 1e-6);
  for (const edge of found) edge.weight = 1 - (edge.d - near) / span;
  edges = found;
}

/** Trace one edge; the caller owns stroke style and dash. */
function edgePath(context, edge) {
  context.beginPath();
  context.moveTo(edge.a.x, edge.a.y);
  context.lineTo(edge.b.x, edge.b.y);
  context.stroke();
}


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
  ink.fillStyle = tint(0.055);
  for (let x = view.width / 2 % 42; x < view.width; x += 42) {
    for (let y = view.height / 2 % 42; y < view.height; y += 42) {
      ink.fillRect(Math.round(x), Math.round(y), 1, 1);
    }
  }

  // Axis rails: the map is a projection with units, so label them.
  ink.font = '9px ui-monospace, monospace';
  ink.fillStyle = tint(0.30);
  ink.strokeStyle = tint(0.14);
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

  // Fifty soft haloes used to sit here. They filled the same space without
  // saying anything; the graph says who is next to whom, and where the
  // projection is lying about it.
  ink.lineWidth = 1;
  for (const edge of edges) {
    ink.strokeStyle = role('edge', Number((0.10 + 0.22 * edge.weight).toFixed(3)));
    ink.setLineDash(edge.shown < FOLDED ? [2, 4] : []);
    edgePath(ink, edge);
  }
  ink.setLineDash([]);
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
    // Its edges brighten, so you can see which presets you are between.
    paint.lineWidth = 1;
    for (const edge of edges) {
      if (edge.a !== sounding && edge.b !== sounding) continue;
      paint.strokeStyle = role('edge', Number((0.38 + 0.42 * edge.weight).toFixed(3)));
      paint.setLineDash(edge.shown < FOLDED ? [2, 4] : []);
      edgePath(paint, edge);
    }
    paint.setLineDash([]);
    for (const radius of [13, 17]) {
      paint.beginPath();
      paint.arc(sounding.x, sounding.y, radius, 0, Math.PI * 2);
      paint.strokeStyle = role('accent', 0.85);
      paint.lineWidth = 1;
      paint.stroke();
    }
  }

  for (const cell of cells) {
    if (cell.test) {
      paint.beginPath();
      paint.arc(cell.x, cell.y, 11, 0, Math.PI * 2);
      paint.strokeStyle = role('test', 0.7);
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
    paint.strokeStyle = role('accent', 0.35);
    paint.setLineDash([2, 3]);
    paint.lineWidth = 1;
    paint.beginPath();
    paint.moveTo(cursor[0], cursor[1]);
    paint.lineTo(voice[0], voice[1]);
    paint.stroke();
    paint.setLineDash([]);
  }
  if (voice) {
    // The last soft halo on the map. A dashed reticle holds the same amount of
    // attention without blurring the hairlines it sits on top of.
    paint.strokeStyle = role('accent', 0.45);
    paint.lineWidth = 1;
    paint.setLineDash([2, 4]);
    paint.beginPath();
    paint.arc(voice[0], voice[1], 17, 0, Math.PI * 2);
    paint.stroke();
    paint.setLineDash([]);
    paint.strokeStyle = role('accent');
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
    paint.strokeStyle = role('accent2', 0.75);
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
    ? role('path', 0.85)
    : role('path', 0.5);
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
    paint.strokeStyle = role('path', 0.8);
    paint.strokeRect(x - 3.5, y - 3.5, 7, 7);
  }

  if (trajectory.playing) {
    const place = pointAt(trajectory.phase);
    if (place) {
      const [x, y] = view.toPixel(place);
      paint.beginPath();
      paint.arc(x, y, 4, 0, Math.PI * 2);
      paint.fillStyle = role('path');
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
    paint.strokeStyle = tint(0.13);
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

  paint.fillStyle = shade(0.72);
  paint.fillRect(left, top, width, height);
  paint.strokeStyle = tint(0.28);
  paint.lineWidth = 1;
  paint.strokeRect(left + 0.5, top + 0.5, width, height);

  paint.font = '9px ui-monospace, monospace';
  paint.textAlign = 'left';
  paint.fillStyle = tint(0.55);
  paint.fillText(`[8D] ${cell.label}`, left + 7, top + 13);

  const rows = cell.pca.length;
  const usable = width - 34;
  for (let axis = 0; axis < rows; axis++) {
    const y = top + 24 + axis * 8;
    paint.fillStyle = tint(0.34);
    paint.fillText(`${axis + 1}`, left + 7, y + 3);
    const middle = left + 20 + usable / 2;
    paint.fillStyle = tint(0.14);
    paint.fillRect(left + 20, y, usable, 1);          // the zero line
    const extent = clamp(cell.pca[axis], -1, 1) * (usable / 2);
    paint.fillStyle = tint(0.85);
    paint.fillRect(Math.min(middle, middle + extent), y - 2, Math.abs(extent), 4);
    paint.fillStyle = tint(0.30);
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
  return scope.source === 'live' ? live.chain : live.fileChain;
}

/* ---------- scope: several views of the same analyser ---------- */

/**
 * Three views of the same signal, stacked: spectrogram, waterfall, loudness.
 *
 * One analyser pass per frame feeds all three, so the cost is the drawing, not
 * the analysis. Each view keeps its own history because they scroll at
 * different rates and would otherwise fight over one buffer.
 */
const scope = {
  analyser: null,
  source: 'silent',
  detail: '',
  running: false,
  freq: null,
  time: null,
  last: 0,
  views: [],
};

// Measured across the atlas: centroids run 90–1033 Hz and 85% rolloff never
// passes 1.4 kHz, so a pad puts nearly everything under ~2 kHz. Drawing up to
// Nyquist spent three quarters of the pane on silence and squeezed every real
// difference into the left edge, which is why the waterfall looked static.
const SCOPE_CEILING = 4000;
const scopeTop = rate => Math.min(SCOPE_CEILING, rate / 2);

const mel = frequency => 2595 * Math.log10(1 + frequency / 700);
const melInverse = value => 700 * (10 ** (value / 2595) - 1);

const SCOPE_VIEWS = [
  {key: 'spectrogram', draw: () => viewSpectrogram},
  {key: 'ridge', draw: () => viewRidge},
  {key: 'loudness', draw: () => viewLoudness},
  {key: 'spectrum', draw: () => viewSpectrum},
  {key: 'wave', draw: () => viewWave},
];
const DEFAULT_SLOTS = ['spectrogram', 'ridge', 'loudness'];

function makeView(slot, key) {
  const canvas = $(`#scope-${slot}`);
  const entry = SCOPE_VIEWS.find(item => item.key === key) || SCOPE_VIEWS[0];
  const tag = canvas.parentElement.querySelector('.scope-tag');
  // The tag is the ASCII half of the label in either language.
  if (tag) tag.textContent = t(`view.${entry.key}`).split(' ').at(-1);
  return {
    slot, key: entry.key, draw: entry.draw(), canvas,
    strip: document.createElement('canvas'),
    width: 0, height: 0, carry: 0,
    rows: null, ridges: [], levels: [],
  };
}

function scopeLayout() {
  const ratio = window.devicePixelRatio || 1;
  for (const view of scope.views) {
    const box = view.canvas.getBoundingClientRect();
    if (box.width < 2 || box.height < 2) continue;    // drawer shut
    view.width = Math.max(80, Math.round(box.width));
    view.height = Math.max(40, Math.round(box.height));
    view.canvas.width = Math.round(box.width * ratio);
    view.canvas.height = Math.round(box.height * ratio);
    view.canvas.getContext('2d').setTransform(ratio, 0, 0, ratio, 0, 0);
    view.strip.width = view.width;
    view.strip.height = view.height;
    const ink = view.strip.getContext('2d');
    ink.fillStyle = BG;
    ink.fillRect(0, 0, view.width, view.height);
    view.rows = null;
    view.ridges = [];
    view.levels = [];
  }
}

function melRows(view, analyser, rate) {
  const bins = analyser.frequencyBinCount;
  const nyquist = rate / 2;
  const top = mel(scopeTop(rate));
  const bottom = mel(40);
  return Array.from({length: view.height}, (_, row) => {
    const high = melInverse(bottom + (top - bottom) * (1 - row / view.height));
    const low = melInverse(bottom + (top - bottom) * (1 - (row + 1) / view.height));
    return [
      Math.max(0, Math.floor(low / nyquist * bins)),
      Math.max(1, Math.min(bins, Math.ceil(high / nyquist * bins))),
    ];
  });
}

/** Redraw the scope's source label in the current language. */
function renderScopeSource() {
  const named = scope.detail
    ? `${scope.detail}${scope.source === 'loop' ? ' \u21bb' : ''}`
    : t(`scope.${scope.source}`);
  $('#scope-source').textContent = named;
}

function useAnalyser(analyser, source, detail = '') {
  scope.analyser = analyser;
  scope.source = source;
  scope.detail = detail || '';
  for (const view of scope.views) view.rows = null;
  renderScopeSource();
  if (!scope.running) {
    scope.running = true;
    scope.last = performance.now();
    requestAnimationFrame(scopeFrame);
  }
}

/**
 * Point the scope at whatever is actually making sound.
 *
 * Stopping a clip with pause() fires no 'ended', so the scope used to stay
 * aimed at a silent file chain: hovering the map made sound again while the
 * display stayed flat. This is called whenever playback state changes, and
 * from the telemetry handler, so it self-heals however it got there.
 */
function restoreScopeSource() {
  const player = $('#player');
  const fileBusy = (player && !player.paused && !player.ended)
    || !!live.loop || !!preview.source;
  if (fileBusy) {
    if (live.fileChain) useAnalyser(live.fileChain.analyser, 'clip', live.playLabel);
    return;
  }
  if (live.connected && live.chain) {
    useAnalyser(live.chain.analyser, 'live');
  } else {
    scope.analyser = null;
    scope.source = 'silent';
    scope.detail = '';
    renderScopeSource();
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
    for (const view of scope.views) view.rows = null;
  }
  analyser.getByteFrequencyData(scope.freq);
  analyser.getByteTimeDomainData(scope.time);
  $('#scope-range').textContent =
    `40Hz–${(scopeTop(rate) / 1000).toFixed(1)}kHz · ${t('scope.stream')} ${(rate / 2000).toFixed(1)}kHz`;

  for (const view of scope.views) {
    if (view.canvas.clientWidth < 2) continue;          // genuinely not on screen
    // Self-heal. Twice now the scope has gone dead because something replaced
    // or resized the views and did not lay them out again; the loop can see
    // that for itself, and a view that is visible but unmeasured is a bug
    // rather than a state worth preserving.
    if (!view.width) scopeLayout();
    if (!view.width) continue;
    const context = view.canvas.getContext('2d');
    context.fillStyle = BG;
    context.fillRect(0, 0, view.width, view.height);
    view.draw(context, view, {rate, elapsed});
  }
  updateCompressorMeter();
}

function viewSpectrogram(context, view, {rate, elapsed}) {
  const strip = view.strip.getContext('2d');
  if (!view.rows) view.rows = melRows(view, scope.analyser, rate);
  view.carry += elapsed * 64;
  const columns = clamp(Math.floor(view.carry), 0, 10);
  view.carry -= columns;
  if (columns > 0) {
    strip.globalCompositeOperation = 'copy';
    strip.drawImage(view.strip, -columns, 0);
    strip.globalCompositeOperation = 'source-over';
    const dropped = live.underruns > live.seenUnderruns;
    live.seenUnderruns = live.underruns;
    for (let row = 0; row < view.height; row++) {
      const [low, high] = view.rows[row];
      let sum = 0;
      for (let bin = low; bin < high; bin++) sum += scope.freq[bin];
      const level = (sum / Math.max(1, high - low) / 255) ** 1.2;
      strip.fillStyle = ramp(level);
      strip.fillRect(view.width - columns, row, columns, 1);
    }
    if (dropped) {
      strip.fillStyle = HOT;
      strip.fillRect(view.width - columns, view.height - 3, Math.max(1, columns), 3);
    }
  }
  context.drawImage(view.strip, 0, 0);
  frequencyGrid(context, view, rate);
}

function viewRidge(context, view, {rate, elapsed}) {
  const bands = 72;
  const bins = scope.freq.length;
  const nyquist = rate / 2;
  const lowest = 40;
  const span = Math.log2(scopeTop(rate) / lowest);
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
  view.carry += elapsed * 22;
  while (view.carry >= 1) {
    view.carry -= 1;
    view.ridges.unshift(line);
    if (view.ridges.length > 34) view.ridges.pop();
  }
  const step = view.height / 38;
  const amplitude = view.height / 3.4;
  context.save();
  context.beginPath();
  context.rect(0, 0, view.width, view.height - 10);
  context.clip();
  for (let index = view.ridges.length - 1; index >= 0; index--) {
    const ridge = view.ridges[index];
    const base = view.height - 8 - index * step;
    const fade = 1 - index / view.ridges.length;
    context.beginPath();
    context.moveTo(0, base);
    for (let band = 0; band < ridge.length; band++) {
      const x = band / (ridge.length - 1) * view.width;
      context.lineTo(x, base - ridge[band] ** 1.3 * amplitude);
    }
    context.lineTo(view.width, base);
    context.closePath();
    context.fillStyle = BG;
    context.fill();
    context.strokeStyle = ramp(0.15 + 0.85 * fade);
    context.lineWidth = 1;
    context.stroke();
  }
  context.restore();
  logFrequencyGrid(context, view, rate, lowest, true);
}

function viewSpectrum(context, view, {rate, elapsed}) {
  const bins = scope.freq.length;
  const nyquist = rate / 2;
  const bars = Math.min(180, Math.floor(view.width / 3));
  if (!view.peaks || view.peaks.length !== bars) view.peaks = new Float32Array(bars);
  const lowest = 40;
  const span = Math.log2(scopeTop(rate) / lowest);
  for (let index = 0; index < bars; index++) {
    const from = lowest * 2 ** (span * index / bars);
    const to = lowest * 2 ** (span * (index + 1) / bars);
    const start = Math.max(0, Math.floor(from / nyquist * bins));
    const end = Math.max(start + 1, Math.min(bins, Math.ceil(to / nyquist * bins)));
    let peak = 0;
    for (let bin = start; bin < end; bin++) peak = Math.max(peak, scope.freq[bin]);
    const level = peak / 255;
    view.peaks[index] = Math.max(level, view.peaks[index] - elapsed * 0.55);
    const x = index / bars * view.width;
    const width = view.width / bars - 1;
    context.fillStyle = ramp(0.2 + 0.8 * level);
    context.fillRect(x, view.height * (1 - level), width, view.height * level);
    context.fillStyle = tint(0.85);
    context.fillRect(x, view.height * (1 - view.peaks[index]) - 1, width, 1);
  }
  logFrequencyGrid(context, view, rate, lowest);
}

/** Triggered on a rising zero crossing, so the wave stands still. */
function viewWave(context, view) {
  const samples = scope.time;
  let trigger = 0;
  for (let index = 1; index < samples.length / 2; index++) {
    if (samples[index - 1] < 128 && samples[index] >= 128) { trigger = index; break; }
  }
  const count = Math.floor(samples.length / 2);
  context.strokeStyle = tint(0.14);
  context.beginPath();
  context.moveTo(0, view.height / 2);
  context.lineTo(view.width, view.height / 2);
  context.stroke();
  context.beginPath();
  for (let index = 0; index < count; index++) {
    const value = (samples[trigger + index] - 128) / 128;
    const x = index / (count - 1) * view.width;
    const y = view.height / 2 - value * view.height * 0.42;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  }
  context.strokeStyle = role('accent2');
  context.lineWidth = 1.2;
  context.stroke();
}

function viewLoudness(context, view, {elapsed}) {
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
  view.carry += elapsed * 40;
  while (view.carry >= 1) {
    view.carry -= 1;
    view.levels.push({
      rms: 20 * Math.log10(Math.max(rms, 1e-5)),
      peak: 20 * Math.log10(Math.max(peak, 1e-5)),
      reduction,
    });
    if (view.levels.length > view.width) view.levels.shift();
  }
  // -72 dB of range put every trace in the top third. The model's output sits
  // around -26 RMS and -17 peak, so the window is the part that carries signal.
  const FLOOR = -42;
  const toY = db => view.height * (1 - clamp((db - FLOOR) / -FLOOR, 0, 1));
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'left';
  for (const db of [-6, -12, -24, -36]) {
    const y = toY(db);
    context.strokeStyle = tint(0.09);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(view.width, y);
    context.stroke();
    context.fillStyle = tint(0.32);
    context.fillText(`${db}`, 4, y - 3);
  }
  const trace = (key, style, width) => {
    context.beginPath();
    view.levels.forEach((value, index) => {
      const x = view.width - view.levels.length + index;
      const y = toY(value[key]);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.strokeStyle = style;
    context.lineWidth = width;
    context.stroke();
  };
  trace('peak', role('accent', 0.55), 1);
  trace('rms', INK, 1.4);
  context.beginPath();
  view.levels.forEach((value, index) => {
    const x = view.width - view.levels.length + index;
    const y = -value.reduction / 24 * view.height;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.strokeStyle = role('accent');
  context.setLineDash([3, 3]);
  context.lineWidth = 1;
  context.stroke();
  context.setLineDash([]);
  const latest = view.levels.at(-1);
  if (latest) {
    context.textAlign = 'right';
    context.fillStyle = INK;
    context.fillText(`RMS ${latest.rms.toFixed(1)}  ${t('level.peak')} ${latest.peak.toFixed(1)} dB`
      + `  ${t('level.squeeze')} ${latest.reduction.toFixed(1)}`, view.width - 6, 12);
  }
}

function frequencyGrid(context, view, rate) {
  const top = mel(scopeTop(rate));
  const bottom = mel(40);
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'left';
  for (const frequency of [100, 250, 500, 1000, 2000]) {
    if (frequency >= scopeTop(rate)) continue;
    const y = view.height * (1 - (mel(frequency) - bottom) / (top - bottom));
    context.strokeStyle = tint(0.10);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(view.width, y);
    context.stroke();
    context.fillStyle = tint(0.34);
    context.fillText(frequency >= 1000 ? `${frequency / 1000}k` : String(frequency), 4, y - 3);
  }
}

function logFrequencyGrid(context, view, rate, lowest, faint = false) {
  const span = Math.log2(scopeTop(rate) / lowest);
  context.font = '9px ui-monospace, monospace';
  context.textAlign = 'center';
  for (const frequency of [100, 250, 500, 1000, 2000]) {
    if (frequency >= scopeTop(rate)) continue;
    const x = Math.log2(frequency / lowest) / span * view.width;
    context.strokeStyle = tint(faint ? 0.05 : 0.09);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, view.height - 10);
    context.stroke();
    context.fillStyle = tint(0.34);
    context.fillText(frequency >= 1000 ? `${frequency / 1000}k` : String(frequency), x, view.height - 2);
  }
}

function updateCompressorMeter() {
  const chain = activeChain();
  if (!chain) return;
  const reductions = chain.compressors.map(item => item.reduction);
  const limited = chain.limiter ? chain.limiter.reduction : 0;
  $('#compMeter').textContent =
    `GR ${reductions.map(value => value.toFixed(0).padStart(3)).join('/')}`
    + (limited < -0.5 ? ` ${t('level.limit')}${limited.toFixed(0)}` : '');
  $('#bands').textContent =
    `${t('band.low')} ${reductions[0].toFixed(1)} dB　${t('band.mid')} ${reductions[1].toFixed(1)} dB`
    + `　${t('band.high')} ${reductions[2].toFixed(1)} dB`;
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
function sendControl(overrides = null) {
  if (audioMode() !== 'live' || !live.connected) return;
  const now = performance.now();
  const since = now - (live.lastSent || 0);
  if (since < 55) {
    if (!live.timer) {
      live.timer = setTimeout(() => { live.timer = 0; sendControl(overrides); }, 55 - since);
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
    live.socket.send(JSON.stringify(
      {type: 'control', seq: ++live.seq, ...currentControls(), ...(overrides || {})}));
  })();
}

function liveSend(message) {
  if (live.socket?.readyState === WebSocket.OPEN) live.socket.send(JSON.stringify(message));
}

/**
 * One small panel that says what is happening, or what you are pointing at.
 *
 * Ableton's info view: hover a control and it explains that control; hover
 * nothing and it reports state. Everything it shows used to be scattered —
 * status in the top right, the sounding preset and the coordinate in two more
 * overlays, a projection note across the middle of the map.
 */
const info = {title: '—', body: '', meta: '', hover: null, prog: ''};

function renderInfo() {
  $('#info-title').textContent = info.hover ? info.hover.title : info.title;
  $('#info-body').textContent = info.hover ? info.hover.body : info.body;
  $('#info-meta').textContent = info.meta;
}

function updateReadout() {
  renderInfo();
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
  const parts = [];
  if (live.connected) {
    parts.push(`${t('meta.plan')} ${live.planMs.toFixed(0)}ms`);
    if (live.rate) parts.push(`${live.rate.toFixed(0)}kbit/s`);
    parts.push(`${t('meta.buffer')} ${live.buffered.toFixed(1)}s`);
    if (live.underruns) parts.push(`${t('meta.underrun')} ${live.underruns}`);
    if (live.dropped) parts.push(`${t('meta.dropped')} ${live.dropped}`);
  }
  parts.push(`PC1 ${coordinate ? coordinate[0].toFixed(2) : '—'}`);
  parts.push(`PC2 ${coordinate ? coordinate[1].toFixed(2) : '—'}`);
  const label = STRINGS[`state.${live.shown}`] ? t(`state.${live.shown}`) : live.shown;
  info.title = `[${live.connected ? 'LIVE' : 'IDLE'}] ${label}`;
  info.meta = parts.join('  ·  ');
  renderInfo();
}

function updateVoice() {
  // Cached mode has no socket, so this cannot be gated on the live connection:
  // what is sounding is whatever last started, from either path.
  if (info.prog) {
    info.body = info.prog;               // a running progression owns this line
    renderInfo();
    return;
  }
  if (!sounding || (!live.connected && audioMode() === 'live')) {
    info.body = '';
    renderInfo();
    return;
  }
  const resting = audioMode() === 'cached'
    ? !preview.source
    : ['idle', 'release', 'offline', 'stopped'].includes(live.shown);
  info.body = sounding
    ? `${sounding.label}${sounding.loner ? t('voice.loner') : ''} `
      + `${resting ? t('voice.resting') : t('voice.sounding')}　${t('voice.component')} ${sounding.component}`
    : '';
  renderInfo();
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
  gateState('connecting'); live.shown = 'connecting';
  updateStatus();
  useAnalyser(live.chain.analyser, 'live');

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
        + `${value.format} · ${t('wire.ceiling')} ${value.kbitPerSecond} kbit/s`;
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
      if (scope.source !== 'live') restoreScopeSource();
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
    live.lifecycle = event.code === 1001 ? 'taken_over' : 'offline';
    // Closing the socket on purpose when switching to cached mode is not an
    // outage, so it must not report one.
    live.shown = audioMode() === 'cached' ? 'cached' : live.lifecycle;
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
  live.shown = 'cached';
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
  setWork(t('work.rendering'));
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
    if (!response.ok) throw new Error(payload.error || `${t('work.renderFailed')} ${response.status}`);
    addTake(payload, sounding ? sounding.label : t('work.freeCoord'));
    setWork(`${payload.seconds.toFixed(1)}s · ${(payload.bytes / 1024).toFixed(0)}KB`);
    play(pickAudio(payload), t('scope.clip'));
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
      ? `${t('take.recorded')} · ${(take.bytes / 1024).toFixed(0)} KB`
      : `${t('take.rendered')} · ${(take.bytes / 1024).toFixed(0)} KB · ${t('level.peak')} ${take.peakDbfs} dBFS`
        + ` · GPU ${(take.renderMs / 1000).toFixed(1)} s`;
    button.addEventListener('click', () => {
      const url = take.recorded ? take.url : pickAudio(take);
      const label = take.recorded ? t('work.recordTake') : take.label;
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
    setWork(t('work.noRecorder'));
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
    button.textContent = t('work.recordChip');
    const blob = new Blob(live.chunks, {type: recorder.mimeType});
    const url = URL.createObjectURL(blob);
    const seconds = (performance.now() - started) / 1000;
    addTake({
      id: url, url, seconds, bytes: blob.size, recorded: true,
      peakDbfs: 0, renderMs: 0,
    }, t('work.recordTake'));
    setWork(`${t('work.recorded')} ${seconds.toFixed(1)}s`);
  };
  const started = performance.now();
  recorder.start();
  live.recorder = recorder;
  button.setAttribute('aria-pressed', 'true');
  button.textContent = t('work.recordingChip');
  setWork(t('work.recording'));
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
    setWork(`${t('work.chainFailed')} ${error.name || error}`);
    return null;
  });
  await live.fileContext?.resume().catch(() => {});
  return chain && live.fileContext?.state === 'running';
}

async function play(url, label = null) {
  const player = $('#player');
  const chain = await filePlayback().catch(error => {
    setWork(`${t('work.chainFailed')} ${error.name || error}`);
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
      ? t('work.blocked')
      : `${t('work.playFailed')} ${error.name || error}`);
    return;
  }
  if (live.fileContext && live.fileContext.state !== 'running') {
    setWork(t('work.suspended'));
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
    setWork(`${t('work.prefetchFail')} ${error.message || error}`);
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
  useAnalyser(chain.analyser, 'preview');
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
      setWork(`${t('work.prefetch')} ${done}/${cells.length}`);
    }
  });
  await Promise.all(workers);
  preview.warming = false;
  if (preview.note === note) {
    setWork(`${t('work.prefetchDone')} ${cells.length}${t('work.prefetchHint')}`);
    if (queue.length === 0) live.shown = 'cached';
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
  setWork(`${t('work.fetching')} ${cell.label}…`);
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
  speed: null,       // per-phase speed profile, see buildSpeedProfile
  sent: 0,           // when the last target was actually emitted
  sentAt: null,      // and where, so the next one can be a real step away
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

/**
 * A speed profile along the stroke, so the walk reads as a hand and not as a
 * turntable.
 *
 * Constant arc-length speed was the obvious thing and the wrong one: it takes
 * hairpins at the same rate as straights, which is the one thing a hand never
 * does. People obey the two-thirds power law when they draw — through a curve
 * of radius R the hand travels at a speed proportional to R^(1/3) — so that is
 * what this computes, from the curvature of the stroke the user actually drew.
 * The ends get an ease as well, because the walk reverses there and hitting a
 * wall at full speed is audible.
 *
 * The profile is normalised by its harmonic mean, so varying the speed does not
 * change how long a lap takes: the 周期 dial still means what it says.
 */
const PATH_SAMPLES = 160;
const EASE_SPAN = 0.12;

function buildSpeedProfile() {
  const count = PATH_SAMPLES;
  const samples = [];
  for (let index = 0; index < count; index++) samples.push(pointAt(index / (count - 1)));
  const ds = trajectory.total / (count - 1) || 1e-6;
  const speed = new Array(count).fill(1);
  for (let index = 1; index < count - 1; index++) {
    const dx = samples[index - 1][0] - 2 * samples[index][0] + samples[index + 1][0];
    const dy = samples[index - 1][1] - 2 * samples[index][1] + samples[index + 1][1];
    const curvature = Math.hypot(dx, dy) / (ds * ds);
    speed[index] = Math.cbrt(1 / Math.max(curvature, 1e-6));
  }
  speed[0] = speed[1];
  speed[count - 1] = speed[count - 2];

  // Normalise against the median rather than the mean, then band it: one
  // scribbled hairpin should colour the walk, not halt it.
  const median = [...speed].sort((a, b) => a - b)[count >> 1] || 1;
  for (let index = 0; index < count; index++) {
    speed[index] = clamp(speed[index] / median, 0.4, 1.7);
  }
  for (let index = 0; index < count; index++) {
    const phase = index / (count - 1);
    const edge = Math.min(phase, 1 - phase) / EASE_SPAN;
    if (edge < 1) speed[index] *= 0.3 + 0.7 * (edge * edge * (3 - 2 * edge));
  }
  let harmonic = 0;
  for (const value of speed) harmonic += 1 / value;
  harmonic /= count;
  for (let index = 0; index < count; index++) speed[index] *= harmonic;
  trajectory.speed = speed;
}

function speedAt(phase) {
  const profile = trajectory.speed;
  if (!profile?.length) return 1;
  const place = clamp(phase, 0, 1) * (profile.length - 1);
  const index = Math.floor(place);
  const next = profile[index + 1] ?? profile[index];
  return profile[index] + (next - profile[index]) * (place - index);
}

/**
 * Move the *sound* along the path, which is a different problem from moving
 * the marker.
 *
 * The runtime does not sweep through latent space. It renders a point and
 * crossfades to it, one plan at a time, from a queue one deep: plans start at
 * most every 0.25 s and each takes as long as it takes (~0.3-0.9 s here).
 * Emitting a target every frame did not make the timbre move faster, it made
 * every render land almost on top of the last one — consecutive points 55 ms
 * apart on the path are nearly the same timbre, so the crossfade had nothing
 * to cross and the walk sounded static.
 *
 * So emit on distance covered, sized to what the pipeline is actually
 * achieving, and set the crossfade to about the gap between steps: one render
 * is still fading up as the next is planned. Stepped underneath, continuous on
 * the ear.
 */
/**
 * Lift a drawn point from the plane into the full 8-D space.
 *
 * A stroke only moves PC1 and PC2. The other six axes stayed wherever the
 * coordinate panel last left them, so a trajectory swept a flat slice of the
 * atlas: measured, a step moved 0.106 in the plane -- a full neighbour spacing
 * there -- but only 0.106 of the 0.827 that separates neighbouring presets in
 * 8-D. Most of what makes two pads sound different was pinned down for the
 * whole walk, which is why the timbre barely moved.
 *
 * So the hidden axes follow the presets the path is passing: an
 * inverse-distance blend of the nearest few, in the plane. The path still owns
 * where you are; the atlas fills in the rest.
 */
const NEIGHBOUR_SPACING = 0.098;
// Every cell contributes, weighted by a Gaussian a few neighbour-spacings
// wide. Taking a hard nearest-three instead made the blend jump each time the
// walk crossed a rank boundary: 8-D travel came out at 41.8 for a span of 3.7,
// eleven times more movement than ground covered, which is jitter, not motion.
const LIFT_BANDWIDTH = NEIGHBOUR_SPACING * 3;

function liftToAtlas(place) {
  const lifted = coordinate.slice();
  lifted[0] = place[0];
  lifted[1] = place[1];
  if (!cells.length) return lifted;
  let total = 0;
  const blend = new Array(8).fill(0);
  for (const cell of cells) {
    const reach = Math.hypot(cell.pca[0] - place[0], cell.pca[1] - place[1]) / LIFT_BANDWIDTH;
    const weight = Math.exp(-reach * reach) + 1e-4;
    total += weight;
    for (let axis = 2; axis < 8; axis++) blend[axis] += cell.pca[axis] * weight;
  }
  for (let axis = 2; axis < 8; axis++) lifted[axis] = blend[axis] / total;
  return lifted;
}

function stepTrajectoryVoice(place, now) {
  const plan = clamp((live.planMs || 300) / 1000, 0.25, 1.4);
  const since = (now - (trajectory.sent || 0)) / 1000;
  if (since < plan * 0.9) return;
  const from = trajectory.sentAt;
  const moved = from ? Math.hypot(place[0] - from[0], place[1] - from[1]) : Infinity;
  // A stride worth rendering, measured against the atlas rather than against
  // the clock: presets sit a median 0.098 apart in the plane a path moves
  // through, so a step shorter than that renders a timbre you have already
  // heard. Gating on the plan cadence instead (what this did first) emitted at
  // exactly the rate the server already managed, which measured as no change
  // at all.
  if (moved < NEIGHBOUR_SPACING * 0.8) return;
  trajectory.sent = now;
  trajectory.sentAt = place.slice();
  // Crossfade for the gap just measured, not longer. A 0.6 s morph across a
  // 0.33 s step means every render is still fading when the next one starts,
  // so the voice never actually arrives anywhere -- a low-pass on the walk.
  sendControl({morphSeconds: clamp(since, 0.5, 5)});
}

function walkTrajectory(now) {
  if (!trajectory.playing) return;
  trajectory.frame = requestAnimationFrame(walkTrajectory);
  const lap = Math.max(0.5, Number($('#lap').value));
  const elapsed = Math.min(0.25, (now - (trajectory.last || now)) / 1000);
  trajectory.last = now;
  // There and back, so a stroke reads as a sweep rather than a jump cut at the
  // end of every lap.
  trajectory.phase += (trajectory.forward ? 1 : -1) * elapsed / lap * speedAt(trajectory.phase);
  if (trajectory.phase >= 1) { trajectory.phase = 1; trajectory.forward = false; }
  if (trajectory.phase <= 0) { trajectory.phase = 0; trajectory.forward = true; }
  const place = pointAt(trajectory.phase);
  if (!place) return;
  coordinate = liftToAtlas(place);
  // The ring glides at frame rate and the crosshair steps along behind it, so
  // the leash between them shows how far the sound is lagging the path. The
  // ring used not to move at all here, which is half of why the walk looked
  // dead even when it was working.
  pointer = place.slice();
  hovered = null;
  syncAxes();
  stepTrajectoryVoice(place, now);
  updateReadout();
  drawMap();
}

function startTrajectory() {
  if (trajectory.points.length < 2) return;
  measurePath();
  buildSpeedProfile();
  trajectory.sent = 0;
  trajectory.sentAt = null;
  trajectory.playing = true;
  trajectory.last = performance.now();
  setWork(`[PATH] ${trajectory.points.length} ${t('path.points')}`
    + ` · ${Number($('#lap').value).toFixed(1)}${t('path.oneWay')}`);
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
    trajectory.speed = null;
    trajectory.sentAt = null;
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
  useAnalyser(chain.analyser, 'loop', label);
  setWork(`${label} ${t('loop.running')} · ${looped.duration.toFixed(1)}${t('loop.perLap')}`);
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
  {key: 'prog.canon', name: '卡农 15634125', degrees: [1, 5, 6, 3, 4, 1, 2, 5]},
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
  select.textContent = '';     // it is rebuilt on a language change, not only once
  PROGRESSIONS.forEach((item, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = progName(item);
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
  setWork(`${progName(item)} ${t('work.rendering')}${onPath ? ` (${t('prog.onPath')})` : ''}`);
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
    if (!response.ok) throw new Error(payload.error || `${t('work.renderFailed')} ${response.status}`);
    addTake(payload, `${item.name} ${noteName(root)}${onPath ? ' ↗' : ''}`);
    setWork(`${progName(item)} · ${payload.voices} ${t('prog.voices')} · ${payload.seconds.toFixed(1)}s`
      + `${onPath ? ` · ${t('prog.onPath')}` : ''}`
      + `${payload.cached ? ` · ${t('prog.cached')}` : ` · GPU ${(payload.renderMs / 1000).toFixed(1)}s`}`);
    // The chord listing used to sit in its own paragraph; it belongs in the
    // info panel now, which is the one place that says what is sounding.
    info.prog = `${progName(item)} ${t('prog.at')} ${noteName(root)}: `
      + chords.map(c => c.map(noteName).join('-')).join('　')
      + `${onPath ? `　${t('prog.onPath')}` : ''}`;
    info.body = info.prog;
    renderInfo();
    const release = payload.take?.release ?? 2.4;
    if (looping()) {
      await playLooping(pickAudio(payload), `${t('prog.progName')} ${progName(item)}`, release);
    } else {
      play(pickAudio(payload), `${t('prog.progName')} ${progName(item)}`);
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
  gateState('stopped');
  live.shown = 'stopped';
  info.prog = '';
  sounding = null;
  updateStatus();
  updateVoice();
  drawMap();
  restoreScopeSource();
  setWork('');
}

/* ---------- panel ---------- */

function syncAxes() {
  document.querySelectorAll('#axes input').forEach((input, index) => {
    input.value = coordinate[index];
    input.nextElementSibling.textContent = coordinate[index].toFixed(2);
  });
}

/** Open or collapse the scope pane, and re-measure everything it moved. */
function setDrawer(shut) {
  $('#board').classList.toggle('drawer-shut', shut);
  $('#drawer').textContent = shut ? '‹ SCOPE' : '›';
  $('#drawer').title = shut ? t('drawer.open') : t('drawer.shut');
  // Both panes changed width; the map and every scope canvas must re-measure.
  requestAnimationFrame(() => { layout(); scopeLayout(); });
}

/** Which view sits in each of the three panes, chosen in settings. */
function buildSlots(rebuildOnly = false) {
  for (const select of document.querySelectorAll('.slot')) {
    for (const option of select.options) option.textContent = t(`view.${option.value}`);
  }
  const slots = ['spectrogram', 'ridge', 'loudness'];   // the three canvas ids
  scope.views = slots.map((slotId, index) => {
    const select = $(`#slot${index}`);
    if (!rebuildOnly && !select.options.length) {
      for (const entry of SCOPE_VIEWS) {
        const option = document.createElement('option');
        option.value = entry.key;
        option.textContent = t(`view.${entry.key}`);
        select.append(option);
      }
      select.value = DEFAULT_SLOTS[index];
      select.addEventListener('change', () => { buildSlots(true); scopeLayout(); });
    }
    return makeView(slotId, select.value);
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
  $('#gates').textContent = `${t('gates.auto')} ${passed}/${total}: `
    + Object.entries(gates).map(([name, value]) => `${value ? '✓' : '✗'} ${name}`).join('　');
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
    try { map.setPointerCapture(event.pointerId); } catch { /* pointer already gone */ }
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
        setWork(t('work.pathShort'));
        trajectory.points = [];
      }
      drawMap();
      return;
    }
    if (trajectory.playing) return;
    moveTo(place(event));
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
    if (on && live.connected) liveSend({type: 'note_off'});
  });
  $('#comp').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
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
  $('#theme').value = document.documentElement.dataset.theme || 'mono';
  $('#theme').addEventListener('change', event => applyTheme(event.target.value));
  $('#lang').value = language;
  $('#lang').addEventListener('change', event => {
    language = languages.includes(event.target.value) ? event.target.value : 'zh';
    applyLanguage();
  });
  $('#drawer').addEventListener('click', () => setDrawer(!$('#board').classList.contains('drawer-shut')));
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
      setWork(t('path.hint'));
    }
  });
  $('#lap').addEventListener('input', event => {
    $('#lapOut').textContent = `${Number(event.target.value).toFixed(1)} s`;
    if (trajectory.playing) {
      setWork(`[PATH] ${trajectory.points.length} ${t('path.points')}`
        + ` · ${Number(event.target.value).toFixed(1)}${t('path.oneWay')}`);
    }
  });
  $('#loop').addEventListener('click', event => {
    const on = event.target.getAttribute('aria-pressed') === 'true';
    event.target.setAttribute('aria-pressed', String(!on));
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
      live.shown = 'cached';
      updateStatus();
      warmPreviews();
    }
  });
  const player = $('#player');
  player.addEventListener('play', () => {
    if ($('#hold').getAttribute('aria-pressed') !== 'true') liveSend({type: 'note_off'});
  });
  player.addEventListener('playing', restoreScopeSource);
  player.addEventListener('ended', restoreScopeSource);
  player.addEventListener('pause', restoreScopeSource);

  window.addEventListener('keydown', event => {
    const key = event.key.toLowerCase();
    if (event.repeat || event.target.matches('input, select') || !(key in KEYS)) return;
    $('#note').value = KEYS[key];
    $('#noteOut').textContent = `${KEYS[key]} · ${noteName(KEYS[key])}`;
    if (live.connected) retrigger();
  });
  // Ableton-style: point at a control, the panel explains that control; point
  // at nothing, it goes back to reporting state.
  document.addEventListener('pointerover', event => {
    const target = event.target.closest?.('[data-info]');
    if (!target) return;
    const [title, body] = (target.dataset.info || '').split('|');
    info.hover = {title: title || '', body: body || ''};
    renderInfo();
  });
  document.addEventListener('pointerout', event => {
    const target = event.target.closest?.('[data-info]');
    if (!target || target.contains(event.relatedTarget)) return;
    info.hover = null;
    renderInfo();
  });

  window.addEventListener('resize', () => { layout(); scopeLayout(); });

  // Surface failures in the page. "my colleague saw some JS errors" is not
  // something anyone can act on; a line in the rail naming the error is.
  window.addEventListener('error', event => {
    setWork(`${t('work.jsError')} ${event.message || event.error}`);
  });
  window.addEventListener('unhandledrejection', event => {
    const reason = event.reason;
    setWork(`${t('work.unhandled')} ${reason?.name || ''} ${reason?.message || reason}`.trim());
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
  buildNeighbourGraph();
  coordinate = (status.defaultPcaNormalized || Array(8).fill(0)).slice();
  buildPanel();
  buildProgressionPicker();
  showGates();
  wire();
  buildSlots();
  // On a narrow window the scope would take almost half the map, so it starts
  // as a spine. It is still there, still one click away.
  setDrawer(window.innerWidth <= 1000);
  applyLanguage();
  layout();
  scopeLayout();
  applyTheme(null);        // read the tokens the stylesheet just resolved
  // The server knows whether it is on this machine or across a tunnel; take its
  // word for it rather than guessing from a hostname that is 127.0.0.1 either way.
  if (status.suggestedMode) {
    $('#mode').value = status.suggestedMode;
  }
  if (status.suggestedBufferSeconds) {
    $('#buffer').value = String(status.suggestedBufferSeconds);
    $('#bufferOut').textContent = `${status.suggestedBufferSeconds.toFixed(1)} s`;
  }
  $('#top-note').textContent = `${status.cuda} · ${status.checkpoint}`;
  $('#curtainNote').textContent = status.profile === 'local'
    ? `${t('curtain.local')} (${status.cuda})${t('curtain.localTail')} ${status.suggestedBufferSeconds} s`
    : t('curtain.remote');
  const explained = status.pcaExplained || [];
  live.lifecycle = 'offline';
  live.shown = 'disconnected';
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
  // "It is silent" and "it will not stop" are both answered by one number.
  outputDb: (() => {
    const node = scope.analyser;
    if (!node) return null;
    const frame = new Float32Array(node.fftSize);
    node.getFloatTimeDomainData(frame);
    let sum = 0;
    for (const value of frame) sum += value * value;
    return Number((20 * Math.log10(Math.sqrt(sum / frame.length) + 1e-9)).toFixed(1));
  })(),
  full: coordinate && coordinate.slice(),
  scopeSource: $('#scope-source').textContent,
  slots: scope.views.map(view => view.key),
  // A view with width 0 has never been laid out, and the frame loop
  // skips it -- which looks exactly like a dead visualiser.
  viewWidths: scope.views.map(view => view.width),
  graph: {
    k: KNN,
    edges: edges.length,
    // Pairs the projection draws far closer together than they really are.
    folded: edges.filter(edge => edge.shown < FOLDED).length,
    shownRange: edges.length
      ? [Math.min(...edges.map(edge => edge.shown)), Math.max(...edges.map(edge => edge.shown))]
        .map(value => Number(value.toFixed(3)))
      : null,
  },
});

boot().catch(error => {
  // This used to write to an element the UI no longer has, so a boot failure
  // threw inside its own handler and hid the reason.
  info.title = `[ERROR] ${t('boot.failed')}`;
  info.body = String(error && (error.stack || error.message) || error).slice(0, 220);
  renderInfo();
  console.error(error);
});
