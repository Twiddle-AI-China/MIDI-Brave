class AtlasPcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.left = [];
    this.right = [];
    this.offset = 0;
    this.frames = 0;
    this.underruns = 0;
    this.active = false;
    this.ticks = 0;
    this.port.onmessage = ({data}) => {
      if (data.type === 'pcm') {
        const pcm = new Float32Array(data.buffer);
        const count = pcm.length / 2;
        const l = new Float32Array(count), r = new Float32Array(count);
        for (let i = 0; i < count; i++) { l[i] = pcm[i * 2]; r[i] = pcm[i * 2 + 1]; }
        this.left.push(l); this.right.push(r); this.frames += count; this.active = true;
      } else if (data.type === 'reset') {
        this.left = []; this.right = []; this.offset = 0; this.frames = 0;
        this.underruns = 0; this.active = false;
      }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0], left = out[0], right = out[1] || out[0];
    for (let i = 0; i < left.length; i++) {
      if (!this.left.length) {
        left[i] = 0; right[i] = 0;
        if (this.active) this.underruns++;
        continue;
      }
      left[i] = this.left[0][this.offset]; right[i] = this.right[0][this.offset];
      this.offset++; this.frames--;
      if (this.offset >= this.left[0].length) {
        this.left.shift(); this.right.shift(); this.offset = 0;
      }
    }
    if (++this.ticks % 16 === 0) this.port.postMessage({type:'buffer', bufferedFrames:this.frames, underruns:this.underruns});
    return true;
  }
}
registerProcessor('atlas-pcm-player', AtlasPcmPlayer);
