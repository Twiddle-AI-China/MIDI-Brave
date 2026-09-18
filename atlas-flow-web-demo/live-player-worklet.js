/**
 * PCM sink for the Atlas Flow live stream.
 *
 * Accepts either interleaved float32 or int16, mono or stereo, and holds
 * silence until `prime` frames have arrived. Over a thin link the prime window
 * is what keeps a roam from stuttering: playback only starts once there is a
 * real cushion, and a starved sink re-primes instead of chattering.
 */
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
    if (this.frames >= this.threshold) this.primed = true;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1] || output[0];
    for (let index = 0; index < left.length; index++) {
      if (!this.primed || !this.queue.length) {
        left[index] = 0;
        right[index] = 0;
        if (this.primed) {
          this.underruns++;
          this.primed = false;
          this.threshold = this.reprime;
        }
        continue;
      }
      const [chunkLeft, chunkRight] = this.queue[0];
      left[index] = chunkLeft[this.offset];
      right[index] = chunkRight[this.offset];
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
      });
    }
    return true;
  }
}

registerProcessor('atlas-live-player', AtlasLivePlayer);
