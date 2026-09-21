/**
 * PCM sink for the Atlas Flow live stream.
 *
 * Accepts interleaved float32 or int16, mono or stereo, at whatever rate the
 * server negotiated, and plays it at whatever rate the output device runs.
 *
 * Resampling here rather than forcing the AudioContext to the wire rate is the
 * whole point. Forcing it -- `new AudioContext({sampleRate: 44100})` on a
 * device that runs at 48000 -- does not avoid a conversion, it just hands the
 * conversion to the browser and the OS, continuously, at a 147:160 ratio, with
 * no way to inspect it. Letting the context come up native and converting once
 * on the way in keeps the whole path visible and works for any device rate.
 *
 * Catmull-Rom over a ring buffer. Cubic rather than linear because it costs
 * almost nothing here, and a ring buffer rather than a queue of chunks because
 * interpolation needs four consecutive samples and they must be allowed to
 * straddle a chunk boundary.
 */

// About 1.5 ms at 44.1 kHz: long enough to kill a step, short enough that a
// genuine gap still reads as a gap rather than as a fade.
const RAMP_SAMPLES = 64;
const RAMP_PER_SAMPLE = 1 / RAMP_SAMPLES;
// ~10 s at 48 kHz. Allocated once; the sink never grows it.
const CAPACITY = 1 << 19;

class AtlasLivePlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.left = new Float32Array(CAPACITY);
    this.right = new Float32Array(CAPACITY);
    this.channels = 2;
    this.dtype = 'float32';
    this.prime = 0;
    this.reprime = 0;
    this.cap = 0;
    this.ratio = 1;
    this.reset();
    this.port.onmessage = ({data}) => {
      if (data.type === 'format') {
        this.channels = data.channels;
        this.dtype = data.dtype;
        this.prime = data.prime;
        // A single late block should cost a short gap, not another full prime.
        this.reprime = data.reprime;
        // The producer fills faster than real time until the first buffer
        // report reaches it, and anything it overshoots by would otherwise sit
        // in the queue as permanent latency. Shed it instead.
        this.cap = data.cap;
        // How many input frames to consume per output frame. `sampleRate` is
        // the context's own rate, which is now always the device's.
        this.ratio = (data.streamRate || sampleRate) / sampleRate;
        this.reset();
      } else if (data.type === 'pcm') {
        this.push(data.buffer);
      } else if (data.type === 'reset') {
        this.reset();
      }
    };
  }

  reset() {
    this.write = 0;         // absolute frame index written
    this.read = 0;          // absolute fractional frame index read
    this.underruns = 0;
    this.dropped = 0;
    this.primed = false;
    this.threshold = this.prime;
    this.ticks = 0;
    this.gain = 0;          // ramps, so starting and starving are not steps
  }

  get available() {
    return this.write - this.read;
  }

  push(buffer) {
    const raw = this.dtype === 'int16' ? new Int16Array(buffer) : new Float32Array(buffer);
    const scale = this.dtype === 'int16' ? 1 / 32767 : 1;
    const count = Math.floor(raw.length / this.channels);
    for (let index = 0; index < count; index++) {
      const slot = (this.write + index) % CAPACITY;
      const value = raw[index * this.channels] * scale;
      this.left[slot] = value;
      this.right[slot] = this.channels === 2 ? raw[index * this.channels + 1] * scale : value;
    }
    this.write += count;
    if (this.cap && this.available > this.cap) {
      // Skip ahead rather than let the overshoot become permanent latency.
      this.read = this.write - this.cap;
      this.dropped++;
    }
    if (this.available >= this.threshold) this.primed = true;
  }

  /** Catmull-Rom through the four frames around `position`. */
  sample(track, position) {
    const base = Math.floor(position);
    const t = position - base;
    const at = offset => track[(((base + offset) % CAPACITY) + CAPACITY) % CAPACITY];
    const p0 = at(-1);
    const p1 = at(0);
    const p2 = at(1);
    const p3 = at(2);
    return p1 + 0.5 * t * (p2 - p0
      + t * (2 * p0 - 5 * p1 + 4 * p2 - p3
      + t * (3 * (p1 - p2) + p3 - p0)));
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1] || output[0];
    for (let index = 0; index < left.length; index++) {
      // Cubic needs one frame of lookahead past the read position, and the
      // fade has to start while data remains -- once the buffer is empty there
      // is nothing left to ramp down.
      const ready = this.available - 2;
      const empty = !this.primed || ready <= 0;
      if (empty && this.primed) {
        this.underruns++;
        this.primed = false;
        this.threshold = this.reprime;
      }
      const ending = this.primed && ready <= RAMP_SAMPLES * this.ratio;
      const want = (empty || ending) ? 0 : 1;
      this.gain = Math.min(1, Math.max(0,
        this.gain + Math.sign(want - this.gain) * RAMP_PER_SAMPLE));
      if (empty || this.gain <= 0) {
        left[index] = 0;
        right[index] = 0;
        continue;
      }
      left[index] = this.sample(this.left, this.read) * this.gain;
      right[index] = this.sample(this.right, this.read) * this.gain;
      this.read += this.ratio;
    }
    if (++this.ticks % 8 === 0) {
      this.port.postMessage({
        type: 'buffer',
        // The read position is fractional now, so this has to be rounded:
        // the runtime validates it as an integer and drops the whole message.
        bufferedFrames: Math.max(0, Math.round(this.available)),
        underruns: this.underruns,
        primed: this.primed,
        dropped: this.dropped,
      });
    }
    return true;
  }
}

registerProcessor('atlas-live-player', AtlasLivePlayer);
