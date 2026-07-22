const CHANNELS = 2;
const PRIMING_FRAMES = 4096;
const STATS_CALLBACK_INTERVAL = 32;

class PcmPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacityFrames = Math.ceil(sampleRate * 1.5);
    this.leftRing = new Float32Array(this.capacityFrames);
    this.rightRing = new Float32Array(this.capacityFrames);
    this.port.onmessage = (event) => {
      if (event.data?.type === 'pcm') {
        this.push(event.data.buffer);
      } else if (event.data?.type === 'reset') {
        this.reset();
      }
    };
    this.reset();
  }

  reset() {
    this.readIndex = 0;
    this.writeIndex = 0;
    this.bufferedFrames = 0;
    this.underruns = 0;
    this.callbackCount = 0;
    this.primed = false;
  }

  push(buffer) {
    if (!(buffer instanceof ArrayBuffer)) {
      return;
    }
    const pcm = new Float32Array(buffer);
    const incomingFrames = Math.floor(pcm.length / CHANNELS);
    if (incomingFrames === 0) {
      return;
    }

    const firstIncomingFrame = Math.max(0, incomingFrames - this.capacityFrames);
    const framesToWrite = incomingFrames - firstIncomingFrame;
    const overflow = Math.max(
      0,
      this.bufferedFrames + framesToWrite - this.capacityFrames,
    );
    if (overflow > 0) {
      this.readIndex = (this.readIndex + overflow) % this.capacityFrames;
      this.bufferedFrames -= overflow;
    }

    for (let frame = firstIncomingFrame; frame < incomingFrames; frame += 1) {
      const sampleIndex = frame * CHANNELS;
      this.leftRing[this.writeIndex] = pcm[sampleIndex];
      this.rightRing[this.writeIndex] = pcm[sampleIndex + 1];
      this.writeIndex = (this.writeIndex + 1) % this.capacityFrames;
    }
    this.bufferedFrames += framesToWrite;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output?.[0];
    const right = output?.[1];
    if (!left || !right) {
      return true;
    }

    left.fill(0);
    right.fill(0);
    if (!this.primed && this.bufferedFrames >= PRIMING_FRAMES) {
      this.primed = true;
    }

    if (this.primed) {
      const framesToRead = Math.min(left.length, this.bufferedFrames);
      for (let frame = 0; frame < framesToRead; frame += 1) {
        left[frame] = this.leftRing[this.readIndex];
        right[frame] = this.rightRing[this.readIndex];
        this.readIndex = (this.readIndex + 1) % this.capacityFrames;
      }
      this.bufferedFrames -= framesToRead;
      if (framesToRead < left.length) {
        this.underruns += 1;
        this.primed = false;
      }
    }

    this.callbackCount += 1;
    if (this.callbackCount % STATS_CALLBACK_INTERVAL === 0) {
      this.port.postMessage({
        type: 'stats',
        bufferedFrames: this.bufferedFrames,
        underruns: this.underruns,
      });
    }
    return true;
  }
}

registerProcessor('pcm-player', PcmPlayerProcessor);
