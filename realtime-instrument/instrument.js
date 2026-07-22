import { clamp, controlFrame, noteName, xyFromPointer } from './instrument-core.js';

const CONTROL_INTERVAL_MS = 30;
const SAMPLE_RATE = 44100;
const state = { x: 0, y: 0, note: 60, velocity: 0.8, seed: 0, seq: 0 };

const canvas = document.querySelector('#xyCanvas');
const context2d = canvas.getContext('2d');
const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const reseedButton = document.querySelector('#reseedButton');
const noteInput = document.querySelector('#noteInput');
const velocityInput = document.querySelector('#velocityInput');
const seedInput = document.querySelector('#seedInput');
const volumeInput = document.querySelector('#volumeInput');
const noteValue = document.querySelector('#noteValue');
const velocityValue = document.querySelector('#velocityValue');
const seedValue = document.querySelector('#seedValue');
const volumeValue = document.querySelector('#volumeValue');
const fieldCoordinates = document.querySelector('#fieldCoordinates');
const connectionStatus = document.querySelector('#connectionStatus');
const statusLamp = document.querySelector('#statusLamp');
const errorLog = document.querySelector('#errorLog');
const cudaValue = document.querySelector('#cudaValue');
const runtimeHashValue = document.querySelector('#runtimeHashValue');
const renderP95Value = document.querySelector('#renderP95Value');
const bufferedValue = document.querySelector('#bufferedValue');
const underrunValue = document.querySelector('#underrunValue');
const midiValue = document.querySelector('#midiValue');
const xyValue = document.querySelector('#xyValue');

const noteKeys = new Map([
  ['a', 60], ['w', 61], ['s', 62], ['e', 63], ['d', 64],
  ['f', 65], ['t', 66], ['g', 67], ['y', 68], ['h', 69],
  ['u', 70], ['j', 71], ['k', 72],
]);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const reportedErrors = new Set();
const trail = [];

let runtimeStatus = null;
let socket = null;
let socketReady = false;
let connectionPromise = null;
let audioContext = null;
let playerNode = null;
let gainNode = null;
let compressorNode = null;
let running = false;
let starting = false;
let pointerId = null;
let controlTimer = null;
let lastControlAt = -Infinity;
let animationFrame = null;
let playerStats = { bufferedFrames: 0, underruns: 0 };

function signed(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

function setConnection(label, tone = 'idle') {
  connectionStatus.textContent = label;
  statusLamp.dataset.tone = tone;
}

function showPersistentError(message) {
  const text = String(message || 'Unknown runtime error');
  if (reportedErrors.has(text)) {
    return;
  }
  reportedErrors.add(text);
  const lower = text.toLowerCase();
  const title = lower.includes('busy')
    ? 'Runtime busy'
    : lower.includes('finite')
      ? 'Non-finite output'
      : 'Runtime error';
  const item = document.createElement('p');
  item.className = 'error-message';
  item.setAttribute('role', 'alert');
  item.textContent = `${title}: ${text}`;
  errorLog.append(item);
  setConnection(title, 'error');
}

function updateTransport() {
  startButton.disabled = running || starting;
  stopButton.disabled = !running;
  reseedButton.disabled = !running;
  startButton.textContent = starting ? 'Starting…' : 'Start';
}

function updateReadouts() {
  const midi = `${noteName(state.note)} · ${state.note}`;
  const xy = `${signed(state.x)} / ${signed(state.y)}`;
  noteValue.textContent = midi;
  velocityValue.textContent = `${Math.round(state.velocity * 100)}%`;
  seedValue.textContent = String(state.seed);
  volumeValue.textContent = `${Math.round(Number(volumeInput.value) * 100)}%`;
  fieldCoordinates.textContent = `X ${signed(state.x)} · Y ${signed(state.y)}`;
  midiValue.textContent = midi;
  xyValue.textContent = xy;
}

function updateRuntimeReadouts(status) {
  runtimeStatus = status;
  cudaValue.textContent = status.cuda || '—';
  runtimeHashValue.textContent = status.runtimeSha256
    ? status.runtimeSha256.slice(0, 12)
    : '—';
  requestCanvasDraw();
}

function updatePlayerReadouts() {
  const rate = audioContext?.sampleRate || runtimeStatus?.sampleRate || SAMPLE_RATE;
  bufferedValue.textContent = `${(playerStats.bufferedFrames / rate * 1000).toFixed(1)} ms`;
  underrunValue.textContent = String(playerStats.underruns);
}

function sendJson(payload) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !socketReady) {
    return false;
  }
  socket.send(JSON.stringify(payload));
  return true;
}

function sendControlNow() {
  controlTimer = null;
  if (!running) {
    return;
  }
  const now = performance.now();
  const remaining = CONTROL_INTERVAL_MS - (now - lastControlAt);
  if (remaining > 0) {
    controlTimer = window.setTimeout(sendControlNow, remaining);
    return;
  }
  const nextSequence = state.seq + 1;
  if (sendJson(controlFrame(nextSequence, state))) {
    state.seq = nextSequence;
    lastControlAt = now;
  }
}

function scheduleControl() {
  if (!running || controlTimer !== null) {
    return;
  }
  const wait = Math.max(0, CONTROL_INTERVAL_MS - (performance.now() - lastControlAt));
  controlTimer = window.setTimeout(sendControlNow, wait);
}

function resetPlayer() {
  playerNode?.port.postMessage({ type: 'reset' });
  playerStats = { bufferedFrames: 0, underruns: 0 };
  updatePlayerReadouts();
}

function stopLocally(label = 'Stopped') {
  running = false;
  starting = false;
  if (controlTimer !== null) {
    clearTimeout(controlTimer);
    controlTimer = null;
  }
  resetPlayer();
  if (audioContext?.state === 'running') {
    void audioContext.suspend().catch((error) => showPersistentError(error.message));
  }
  updateTransport();
  setConnection(label, socketReady ? 'ready' : 'idle');
}

function stopInstrument(label = 'Stopped') {
  sendJson({ type: 'stop' });
  stopLocally(label);
}

function handlePlayerStats(payload) {
  const bufferedFrames = Number(payload.bufferedFrames);
  const underruns = Number(payload.underruns);
  if (!Number.isFinite(bufferedFrames) || !Number.isFinite(underruns)) {
    showPersistentError('audio worklet returned non-finite statistics');
    return;
  }
  playerStats = {
    bufferedFrames: Math.max(0, Math.trunc(bufferedFrames)),
    underruns: Math.max(0, Math.trunc(underruns)),
  };
  updatePlayerReadouts();
  if (running) {
    sendJson({
      type: 'buffer',
      bufferedFrames: playerStats.bufferedFrames,
      underruns: playerStats.underruns,
    });
  }
}

async function ensureAudioGraph() {
  if (audioContext) {
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    return;
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error('AudioWorklet playback is not supported by this browser');
  }

  const nextContext = new AudioContextClass({
    sampleRate: SAMPLE_RATE,
    latencyHint: 'interactive',
  });
  audioContext = nextContext;
  try {
    if (nextContext.state === 'suspended') {
      await nextContext.resume();
    }
    await nextContext.audioWorklet.addModule(
      new URL('./pcm-player-worklet.js', import.meta.url),
    );
    playerNode = new AudioWorkletNode(nextContext, 'pcm-player', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    gainNode = nextContext.createGain();
    gainNode.gain.value = 0.2;
    compressorNode = nextContext.createDynamicsCompressor();
    playerNode.connect(gainNode).connect(compressorNode).connect(nextContext.destination);
    playerNode.port.onmessage = (event) => {
      if (event.data?.type === 'stats') {
        handlePlayerStats(event.data);
      }
    };
    gainNode.gain.value = clamp(volumeInput.value, 0, 1);
  } catch (error) {
    playerNode = null;
    gainNode = null;
    compressorNode = null;
    audioContext = null;
    await nextContext.close();
    throw error;
  }
}

function websocketUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/runtime`;
}

function handleTelemetry(payload) {
  const p95 = Number(payload.renderP95Ms);
  if (!Number.isFinite(p95)) {
    showPersistentError('runtime telemetry contained a non-finite render time');
    return;
  }
  renderP95Value.textContent = `${p95.toFixed(1)} ms`;
}

function forwardPcm(buffer) {
  if (!running || !playerNode || !(buffer instanceof ArrayBuffer)) {
    return;
  }
  playerNode.port.postMessage({ type: 'pcm', buffer }, [buffer]);
}

function connectRuntime() {
  if (socket?.readyState === WebSocket.OPEN && socketReady) {
    return Promise.resolve(socket);
  }
  if (connectionPromise) {
    return connectionPromise;
  }

  setConnection('Connecting runtime', 'idle');
  connectionPromise = new Promise((resolve, reject) => {
    const candidate = new WebSocket(websocketUrl());
    candidate.binaryType = 'arraybuffer';
    socket = candidate;
    socketReady = false;
    let settled = false;

    const rejectOnce = (error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };

    candidate.addEventListener('message', (event) => {
      if (event.data instanceof ArrayBuffer) {
        forwardPcm(event.data);
        return;
      }

      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        const error = new Error('runtime sent invalid telemetry JSON');
        showPersistentError(error.message);
        rejectOnce(error);
        return;
      }

      if (payload.type === 'ready') {
        socketReady = true;
        updateRuntimeReadouts(payload);
        if (!settled) {
          settled = true;
          resolve(candidate);
        }
        if (!running && !starting) {
          setConnection('Runtime ready', 'ready');
        }
      } else if (payload.type === 'telemetry') {
        handleTelemetry(payload);
      } else if (payload.type === 'error') {
        const error = new Error(payload.message || 'runtime rejected the session');
        showPersistentError(error.message);
        rejectOnce(error);
        stopLocally('Start required');
      }
    });

    candidate.addEventListener('error', () => {
      const error = new Error('runtime WebSocket connection failed');
      showPersistentError(error.message);
      rejectOnce(error);
    });

    candidate.addEventListener('close', () => {
      if (socket === candidate) {
        socket = null;
        socketReady = false;
      }
      rejectOnce(new Error('runtime WebSocket closed before it was ready'));
      if (running || starting) {
        showPersistentError('runtime WebSocket closed');
        stopLocally('Start required');
      }
    });
  }).finally(() => {
    connectionPromise = null;
  });

  return connectionPromise;
}

async function startInstrument() {
  if (running || starting || document.hidden) {
    return;
  }
  starting = true;
  updateTransport();
  setConnection('Starting audio', 'idle');
  try {
    // This call constructs AudioContext synchronously inside the click gesture.
    await ensureAudioGraph();
    await connectRuntime();
    resetPlayer();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    const started = sendJson({
      type: 'start',
      x: state.x,
      y: state.y,
      note: state.note,
      velocity: state.velocity,
      seed: state.seed,
    });
    if (!started) {
      throw new Error('runtime was not ready to start');
    }
    running = true;
    starting = false;
    lastControlAt = -Infinity;
    updateTransport();
    setConnection('Live', 'live');
  } catch (error) {
    showPersistentError(error.message);
    stopLocally('Start required');
  }
}

function setStateXY(x, y) {
  state.x = clamp(x, -1, 1);
  state.y = clamp(y, -1, 1);
  if (!reducedMotion.matches) {
    trail.push({ x: state.x, y: state.y, time: performance.now() });
    if (trail.length > 18) {
      trail.shift();
    }
  }
  updateReadouts();
  requestCanvasDraw();
  scheduleControl();
}

function updateXYFromPointer(event) {
  const xy = xyFromPointer(event.clientX, event.clientY, canvas.getBoundingClientRect());
  setStateXY(xy.x, xy.y);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * scale));
  const height = Math.max(1, Math.round(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context2d.setTransform(scale, 0, 0, scale, 0, 0);
  return { width: rect.width, height: rect.height };
}

function canvasPoint(point, width, height) {
  return {
    x: (point.x + 1) * 0.5 * width,
    y: (1 - point.y) * 0.5 * height,
  };
}

function drawField(now = performance.now()) {
  const { width, height } = resizeCanvas();
  context2d.clearRect(0, 0, width, height);
  context2d.fillStyle = '#101d2d';
  context2d.fillRect(0, 0, width, height);

  context2d.lineWidth = 1;
  for (let step = 1; step < 8; step += 1) {
    const x = width * step / 8;
    const y = height * step / 8;
    context2d.strokeStyle = step === 4
      ? 'rgba(241, 179, 91, 0.32)'
      : 'rgba(142, 220, 199, 0.09)';
    context2d.beginPath();
    context2d.moveTo(x, 0);
    context2d.lineTo(x, height);
    context2d.moveTo(0, y);
    context2d.lineTo(width, y);
    context2d.stroke();
  }

  const points = runtimeStatus?.plane?.points || [];
  context2d.fillStyle = 'rgba(142, 220, 199, 0.52)';
  for (const point of points) {
    if (!Array.isArray(point) || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) {
      continue;
    }
    const position = canvasPoint({ x: point[0], y: point[1] }, width, height);
    context2d.beginPath();
    context2d.arc(position.x, position.y, 2.2, 0, Math.PI * 2);
    context2d.fill();
  }

  if (!reducedMotion.matches && trail.length > 1) {
    const cutoff = now - 720;
    while (trail.length && trail[0].time < cutoff) {
      trail.shift();
    }
    for (let index = 1; index < trail.length; index += 1) {
      const from = canvasPoint(trail[index - 1], width, height);
      const to = canvasPoint(trail[index], width, height);
      const alpha = clamp((trail[index].time - cutoff) / 720, 0, 1) * 0.7;
      context2d.strokeStyle = `rgba(241, 179, 91, ${alpha})`;
      context2d.lineWidth = 1.5;
      context2d.beginPath();
      context2d.moveTo(from.x, from.y);
      context2d.lineTo(to.x, to.y);
      context2d.stroke();
    }
  }

  const playhead = canvasPoint(state, width, height);
  context2d.strokeStyle = '#f1b35b';
  context2d.lineWidth = 1;
  context2d.beginPath();
  context2d.moveTo(playhead.x - 13, playhead.y);
  context2d.lineTo(playhead.x + 13, playhead.y);
  context2d.moveTo(playhead.x, playhead.y - 13);
  context2d.lineTo(playhead.x, playhead.y + 13);
  context2d.stroke();
  context2d.fillStyle = '#e8eef2';
  context2d.beginPath();
  context2d.arc(playhead.x, playhead.y, 3, 0, Math.PI * 2);
  context2d.fill();
}

function animateCanvas(now) {
  drawField(now);
  animationFrame = requestAnimationFrame(animateCanvas);
}

function requestCanvasDraw() {
  if (reducedMotion.matches) {
    drawField();
  }
}

function configureCanvasMotion() {
  if (reducedMotion.matches) {
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    trail.length = 0;
    drawField();
  } else if (animationFrame === null) {
    animationFrame = requestAnimationFrame(animateCanvas);
  }
}

async function loadRuntimeStatus() {
  try {
    const response = await fetch('/api/runtime-status', {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`runtime status returned HTTP ${response.status}`);
    }
    updateRuntimeReadouts(await response.json());
    setConnection('Runtime ready', 'ready');
  } catch (error) {
    showPersistentError(error.message);
  }
}

canvas.addEventListener('pointerdown', (event) => {
  pointerId = event.pointerId;
  canvas.setPointerCapture(pointerId);
  canvas.focus({ preventScroll: true });
  updateXYFromPointer(event);
});

canvas.addEventListener('pointermove', (event) => {
  if (event.pointerId === pointerId && canvas.hasPointerCapture(pointerId)) {
    updateXYFromPointer(event);
  }
});

function releasePointer(event) {
  if (event.pointerId !== pointerId) {
    return;
  }
  if (canvas.hasPointerCapture(pointerId)) {
    canvas.releasePointerCapture(pointerId);
  }
  pointerId = null;
}

canvas.addEventListener('pointerup', releasePointer);
canvas.addEventListener('pointercancel', releasePointer);

canvas.addEventListener('keydown', (event) => {
  const delta = event.shiftKey ? 0.01 : 0.05;
  const movement = {
    ArrowLeft: [-delta, 0],
    ArrowRight: [delta, 0],
    ArrowDown: [0, -delta],
    ArrowUp: [0, delta],
  }[event.key];
  if (!movement) {
    return;
  }
  event.preventDefault();
  setStateXY(state.x + movement[0], state.y + movement[1]);
});

startButton.addEventListener('click', startInstrument);
stopButton.addEventListener('click', () => stopInstrument());
reseedButton.addEventListener('click', () => {
  if (running) {
    sendJson({ type: 'reseed', seed: state.seed });
  }
});

noteInput.addEventListener('input', () => {
  state.note = Math.round(clamp(noteInput.value, 21, 109));
  updateReadouts();
  scheduleControl();
});

velocityInput.addEventListener('input', () => {
  state.velocity = clamp(velocityInput.value, 0, 1);
  updateReadouts();
  scheduleControl();
});

seedInput.addEventListener('input', () => {
  const seed = Number(seedInput.value);
  if (Number.isFinite(seed)) {
    state.seed = Math.max(0, Math.trunc(seed));
    seedValue.textContent = String(state.seed);
  }
});

volumeInput.addEventListener('input', () => {
  const volume = clamp(volumeInput.value, 0, 1);
  volumeValue.textContent = `${Math.round(volume * 100)}%`;
  if (gainNode && audioContext) {
    gainNode.gain.setTargetAtTime(volume, audioContext.currentTime, 0.01);
  }
});

window.addEventListener('keydown', (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) {
    return;
  }
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) {
    return;
  }
  const note = noteKeys.get(event.key.toLowerCase());
  if (note === undefined) {
    return;
  }
  event.preventDefault();
  state.note = note;
  noteInput.value = String(note);
  updateReadouts();
  scheduleControl();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopInstrument('Start required');
  }
});

window.addEventListener('pagehide', () => {
  sendJson({ type: 'stop' });
  socket?.close();
});

const resizeObserver = new ResizeObserver(requestCanvasDraw);
resizeObserver.observe(canvas);
reducedMotion.addEventListener('change', configureCanvasMotion);

updateReadouts();
updatePlayerReadouts();
updateTransport();
configureCanvasMotion();
void loadRuntimeStatus();
