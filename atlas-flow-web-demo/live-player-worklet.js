/**
 * PCM sink for the Atlas Flow live stream.
 *
 * Accepts either interleaved float32 or int16, mono or stereo, and holds
 * silence until `prime` frames have arrived. Over a thin link the prime window
 * is what keeps a roam from stuttering: playback only starts once there is a
 * real cushion, and a starved sink re-primes instead of chattering.
 */
// About 1.5 ms at 44.1 kHz: long enough to kill the step, short enough that a
// genuine gap still reads as a gap rather than as a fade.
const RAMP_SAMPLES = 64;
const RAMP_PER_SAMPLE = 1 / RAMP_SAMPLES;

class AtlasLivePlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.reset();
    this.channels = 2;
    this.dtype = 'float32';
    this.prime = 0;
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
        this.reset();
      } else if (data.type === 'pcm') {
        this.push(data.buffer);
      } else if (data.type === 'reset') {
        this.reset();
      }
    };
  }

  reset() {
    this.queue = [];
    this.offset = 0;
    this.frames = 0;
    this.underruns = 0;
    this.primed = false;
    this.threshold = this.prime;
    this.ticks = 0;
    this.gain = 0;          // ramps, so starting and starving are not steps
  }

  push(buffer) {
    const raw = this.dtype === 'int16' ? new Int16Array(buffer) : new Float32Array(buffer);
    const scale = this.dtype === 'int16' ? 1 / 32767 : 1;
    const count = raw.length / this.channels;
    const left = new Float32Array(count);
    const right = new Float32Array(count);
    for (let index = 0; index < count; index++) {
      left[index] = raw[index * this.channels] * scale;
      right[index] = this.channels === 2 ? raw[index * this.channels + 1] * scale : left[index];
    }
    this.queue.push([left, right]);
    this.frames += count;
    while (this.cap && this.frames > this.cap && this.queue.length > 1) {
      const [dropped] = this.queue.shift();
      this.frames -= dropped.length - this.offset;
      this.offset = 0;
      this.dropped = (this.dropped || 0) + 1;
    }
    if (this.frames >= this.threshold) this.primed = true;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1] || output[0];
    for (let index = 0; index < left.length; index++) {
      const empty = !this.primed || !this.queue.length;
      if (empty && this.primed) {
        this.underruns++;
        this.primed = false;
        this.threshold = this.reprime;
      }
      // Cutting to zero mid-waveform is a full-scale step, and a run of those
      // is heard as a buzz rather than as a gap -- which is what a cold start
      // produces, because the first plans are several times slower than the
      // warm ones and the queue keeps running dry.
      //
      // The fade has to START while there is still data to fade: once the
      // queue is empty there is nothing left to ramp, which is why fading only
      // on the way back up removed just half the steps.
      const ending = this.primed && this.frames <= RAMP_SAMPLES;
      const want = (empty || ending) ? 0 : 1;
      this.gain = Math.min(1, Math.max(0,
        this.gain + Math.sign(want - this.gain) * RAMP_PER_SAMPLE));
      if (empty || this.gain <= 0) {
        left[index] = 0;
        right[index] = 0;
        continue;
      }
      const [chunkLeft, chunkRight] = this.queue[0];
      left[index] = chunkLeft[this.offset] * this.gain;
      right[index] = chunkRight[this.offset] * this.gain;
      this.offset++;
      this.frames--;
      if (this.offset >= chunkLeft.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (++this.ticks % 8 === 0) {
      this.port.postMessage({
        type: 'buffer',
        bufferedFrames: this.frames,
        underruns: this.underruns,
        primed: this.primed,
        dropped: this.dropped || 0,
      });
    }
    return true;
  }
}

registerProcessor('atlas-live-player', AtlasLivePlayer);
