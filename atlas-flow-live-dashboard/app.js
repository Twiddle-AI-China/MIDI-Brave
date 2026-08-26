const $ = (selector) => document.querySelector(selector);
const gateNames = {
  midi_following_95pct:'MIDI 跟随率 ≥ 95%', dynamic_median_5pct:'动态轨迹中位收益 ≥ 5%',
  dynamic_benefit_70pct:'动态轨迹受益样本 ≥ 70%', silence_le_1pct:'静音失败率 ≤ 1%',
  tail_drift_le_12db:'尾段 RMS 漂移 ≤ 12 dB', boundary_ratio_le_4:'边界突变比 ≤ 4',
  rms_median_le_6db:'RMS 误差中位数 ≤ 6 dB', rms_p90_le_12db:'RMS 误差 P90 ≤ 12 dB',
  deterministic:'确定性复现'
};
const modeLabels = {source:'SOURCE',dynamic:'DYNAMIC',static:'STATIC',flow:'FLOW'};

let runtime = null, points = [], edges = [], ws = null, seq = 0;
let pca = Array(8).fill(0), audiblePca = Array(8).fill(0), projectedPca = Array(8).fill(0);
let runtimeInitialized = false, heldKeys = [], controlTimer = null;
let audioContext = null, player = null, fallback = null;

async function loadEvaluation() {
  const response = await fetch('/api/evaluation', {cache:'no-store'});
  const report = await response.json();
  const complete = report.pipeline_state === 'complete';
  const passed = complete && report.passed === true;
  const badge = $('#qualityBadge');
  badge.textContent = !complete ? '评估尚未完成' : passed ? '自动评估通过' : 'NO-GO · 门禁未通过';
  badge.className = `badge ${passed ? 'good' : complete ? 'bad' : 'pending'}`;
  $('#qualityNotice').className = `notice ${passed ? 'good' : complete ? 'bad' : 'warning'}`;
  $('#qualityNotice').textContent = passed
    ? 'Pad 自动硬门禁已通过；四类别、主观听评与长期实时稳定性仍需单独完成。'
    : complete ? '当前 Pad 阶段结果为 NO-GO：页面保留失败项和完整 test 统计，不将阶段性 Demo 表述为论文就绪。'
    : '评估流水线尚未产出完整报告；实时演奏能力与离线质量门禁必须分开解读。';
  $('#checkpoint').textContent = report.checkpoint || 'pending';
  const gates = $('#gates'); gates.replaceChildren();
  Object.entries(report.gates || {}).forEach(([name, value]) => {
    const row = document.createElement('div'); row.className = 'gate';
    row.innerHTML = `<strong>${gateNames[name] || name}</strong><span class="${value?'pass':'fail'}">${value?'PASS':'FAIL'}</span>`;
    gates.append(row);
  });
  if (!gates.children.length) gates.innerHTML = '<p class="muted">尚无可审核门禁。</p>';
  const limitations = $('#limitations'); limitations.replaceChildren();
  (report.limitations || []).forEach(text => { const li=document.createElement('li'); li.textContent=text; limitations.append(li); });
  const rows = report.audition?.rows || report.rows || [];
  renderAudition(rows);
  const population = report.population?.records;
  $('#sampleCount').textContent = `${rows.length} 组展示 / ${population || rows.length} 组指标`;
}

function renderAudition(rows) {
  const root = $('#samples'); root.replaceChildren();
  const grouped = new Map();
  rows.forEach(row => {
    if (!grouped.has(row.preset_id)) grouped.set(row.preset_id, []);
    grouped.get(row.preset_id).push(row);
  });
  grouped.forEach((presetRows, preset) => {
    presetRows.sort((a,b) => a.note-b.note);
    const card = document.createElement('article'); card.className='sample';
    card.innerHTML = `<div class="sample-head"><div class="sample-title"><b>${preset}</b><small>held-out test preset</small></div><select aria-label="选择 MIDI 音高"></select></div><div class="mode-buttons"></div><audio controls preload="none"></audio><div class="sample-metric"><span class="mode-caption">真实参考</span><span class="metrics"></span></div>`;
    const select = card.querySelector('select');
    presetRows.forEach(row => { const option=document.createElement('option');option.value=String(row.note);option.textContent=`MIDI ${row.note}`;select.append(option); });
    const buttons = card.querySelector('.mode-buttons'), audio = card.querySelector('audio');
    let activeMode='source';
    Object.keys(modeLabels).forEach(mode => { const button=document.createElement('button');button.textContent=modeLabels[mode];button.dataset.mode=mode;if(mode===activeMode)button.classList.add('active');buttons.append(button); });
    const currentRow = () => presetRows.find(row => row.note === +select.value) || presetRows[0];
    const update = (mode, preserve=true) => {
      const row=currentRow(), wasPlaying=!audio.paused, position=preserve?audio.currentTime:0;
      activeMode=mode; buttons.querySelectorAll('button').forEach(button=>button.classList.toggle('active',button.dataset.mode===mode));
      audio.pause(); audio.src=`/evaluation/${row.audio[mode]}`; audio.load();
      audio.addEventListener('loadedmetadata',()=>{audio.currentTime=Math.min(position,Math.max(0,audio.duration-.01));if(wasPlaying)audio.play().catch(()=>{});},{once:true});
      const benefit=Number.isFinite(row.dynamic_improvement_fraction)?`${(row.dynamic_improvement_fraction*100).toFixed(1)}%`:'—';
      card.querySelector('.mode-caption').textContent={source:'真实参考',dynamic:'编码动态轨迹重建',static:'时间均值轨迹消融',flow:'无 Source 轨迹的 Flow 预测'}[mode];
      card.querySelector('.metrics').textContent=`Dynamic 收益 ${benefit} · Flow RMS ${Number(row.flow_rms_error_db).toFixed(1)} dB`;
    };
    buttons.addEventListener('click',event=>{const button=event.target.closest('button');if(button)update(button.dataset.mode,true);});
    select.addEventListener('change',()=>update(activeMode,false));
    audio.addEventListener('play',()=>document.querySelectorAll('#samples audio').forEach(other=>{if(other!==audio)other.pause();}));
    root.append(card); update('source',false);
  });
  if (!root.children.length) root.innerHTML='<p class="muted">试听材料尚未生成。</p>';
}

function buildPcaSliders() {
  const root=$('#pcaSliders'); root.replaceChildren();
  for(let index=0;index<8;index++){
    const row=document.createElement('div'); row.className='pca-row';
    row.innerHTML=`<label for="pc${index+1}">PC${index+1}</label><input id="pc${index+1}" type="range" min="-1" max="1" step="0.01"><output class="target">0.00</output><output class="audible">0.00</output>`;
    const input=row.querySelector('input'); input.value=String(pca[index]);
    input.addEventListener('input',()=>{pca[index]=+input.value;updatePcaReadouts();drawAtlas();scheduleControl();});
    root.append(row);
  }
  updatePcaReadouts();
}

function updatePcaReadouts() {
  document.querySelectorAll('.pca-row').forEach((row,index)=>{
    row.querySelector('input').value=String(pca[index]);
    row.querySelector('.target').textContent=pca[index].toFixed(2);
    row.querySelector('.audible').textContent=audiblePca[index].toFixed(2);
  });
}

function drawAtlas() {
  const canvas=$('#atlas'),ctx=canvas.getContext('2d'),width=canvas.width,height=canvas.height,pad=30;
  ctx.clearRect(0,0,width,height);ctx.strokeStyle='#1d2d3c';ctx.lineWidth=1;
  for(let i=1;i<6;i++){ctx.beginPath();ctx.moveTo(pad,(height-2*pad)*i/6+pad);ctx.lineTo(width-pad,(height-2*pad)*i/6+pad);ctx.stroke();}
  const xy=value=>[pad+(value[0]+1)/2*(width-2*pad),height-pad-(value[1]+1)/2*(height-2*pad)];
  ctx.strokeStyle='#405067';ctx.globalAlpha=.35;
  edges.forEach(([left,right])=>{if(!points[left]||!points[right])return;const a=xy(points[left].pcaNormalized||[points[left].x,points[left].y]),b=xy(points[right].pcaNormalized||[points[right].x,points[right].y]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();});
  ctx.globalAlpha=.82;points.forEach(point=>{const [x,y]=xy(point.pcaNormalized||[point.x,point.y]);ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle=`hsl(${(point.component*61)%360} 70% 68%)`;ctx.fill();});
  ctx.globalAlpha=1;const [tx,ty]=xy(pca),[ax,ay]=xy(audiblePca);
  ctx.beginPath();ctx.arc(tx,ty,12,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
  ctx.beginPath();ctx.arc(ax,ay,6,0,Math.PI*2);ctx.fillStyle='#54d69c';ctx.fill();ctx.strokeStyle='#07130e';ctx.lineWidth=2;ctx.stroke();
}

async function checkRuntime() {
  try {
    const response=await fetch('/api/runtime-status',{cache:'no-store'}),value=await response.json();
    if(!response.ok||value.ok===false)throw new Error(value.message||'GPU runtime offline');
    runtime=value;points=value.points||[];edges=value.edges||[];
    if(!runtimeInitialized){pca=(value.defaultPcaNormalized||Array(8).fill(0)).slice();audiblePca=pca.slice();projectedPca=pca.slice();runtimeInitialized=true;buildPcaSliders();}
    drawAtlas();$('#runtimeBadge').textContent=`在线 · ${value.cuda}`;$('#runtimeBadge').className='badge good';
    if(!ws||ws.readyState!==WebSocket.OPEN){$('#start').disabled=false;$('#liveState').textContent='就绪';}
  }catch(error){$('#runtimeBadge').textContent='GPU 服务离线';$('#runtimeBadge').className='badge bad';$('#liveState').textContent=error.message;if(!ws)$('#start').disabled=true;}
}

function controls(){return{pcaNormalized:pca.slice(),note:+$('#note').value,velocity:+$('#velocity').value,temperature:+$('#temperature').value,morphSeconds:+$('#morph').value};}
function send(value){if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(value));}

async function setupAudio(){
  if(audioContext){await audioContext.resume();return;}
  audioContext=new AudioContext({sampleRate:runtime.sampleRate,latencyHint:'interactive'});
  try{
    await audioContext.audioWorklet.addModule('/pcm-player-worklet.js');
    player=new AudioWorkletNode(audioContext,'atlas-pcm-player',{outputChannelCount:[2]});player.connect(audioContext.destination);
    player.port.onmessage=({data})=>{if(data.type==='buffer'){$('#underruns').textContent=data.underruns;send({type:'buffer',bufferedFrames:data.bufferedFrames,underruns:data.underruns});}};
  }catch(_error){
    const queue=[];let offset=0,underruns=0;fallback=audioContext.createScriptProcessor(2048,0,2);
    fallback.onaudioprocess=event=>{const l=event.outputBuffer.getChannelData(0),r=event.outputBuffer.getChannelData(1);for(let i=0;i<l.length;i++){if(!queue.length){l[i]=r[i]=0;underruns++;continue;}l[i]=queue[0][offset*2];r[i]=queue[0][offset*2+1];if(++offset>=queue[0].length/2){queue.shift();offset=0;}}const buffered=queue.reduce((n,x)=>n+x.length/2,0)-offset;$('#underruns').textContent=underruns;send({type:'buffer',bufferedFrames:Math.max(0,buffered),underruns});};
    fallback.push=buffer=>queue.push(new Float32Array(buffer));fallback.reset=()=>{queue.length=0;offset=0;underruns=0};fallback.connect(audioContext.destination);
  }
}
function queuePcm(buffer){if(player)player.port.postMessage({type:'pcm',buffer},[buffer]);else fallback?.push(buffer);}
function resetAudio(){if(player)player.port.postMessage({type:'reset'});fallback?.reset();}

async function connectAndStart(){
  await setupAudio();resetAudio();if(ws)ws.close();
  const protocol=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(`${protocol}//${location.host}/runtime`);ws.binaryType='arraybuffer';
  $('#liveState').textContent='连接 GPU…';$('#start').disabled=true;
  ws.onmessage=event=>{
    if(event.data instanceof ArrayBuffer){queuePcm(event.data);return;}
    const value=JSON.parse(event.data);
    if(value.type==='ready'){send({type:'start',seed:+$('#seed').value,...controls()});$('#liveState').textContent='后台生成首段轨迹…';}
    if(value.type==='telemetry'){
      audiblePca=(value.audiblePcaNormalized||audiblePca).slice();projectedPca=(value.projectedPcaNormalized||projectedPca).slice();updatePcaReadouts();drawAtlas();
      $('#liveState').textContent=value.lifecycle;$('#planMode').textContent=value.planMode;$('#planMs').textContent=`${Number(value.planMs).toFixed(0)} ms`;$('#component').textContent=value.component;$('#planPending').textContent=value.planPending?'等待/生成':'已同步';
      if(value.lastError)$('#liveState').textContent=`继续旧音色 · ${value.lastError}`;
    }
    if(value.type==='error')$('#liveState').textContent=`当前声音继续 · ${value.message}`;
  };
  ws.onopen=()=>{$('#stop').disabled=false;$('#noteOff').disabled=false;$('#reseed').disabled=false;};
  ws.onclose=()=>{$('#liveState').textContent='已断开';$('#start').disabled=false;$('#stop').disabled=true;$('#noteOff').disabled=true;$('#reseed').disabled=true;heldKeys=[];};
}

function scheduleControl(){clearTimeout(controlTimer);controlTimer=setTimeout(()=>send({type:'control',seq:++seq,...controls()}),50);}
const canvas=$('#atlas');
function pointFromEvent(event){const box=canvas.getBoundingClientRect();pca[0]=Math.max(-1,Math.min(1,(event.clientX-box.left)/box.width*2-1));pca[1]=Math.max(-1,Math.min(1,1-(event.clientY-box.top)/box.height*2));updatePcaReadouts();drawAtlas();scheduleControl();}
canvas.addEventListener('pointerdown',event=>{canvas.setPointerCapture(event.pointerId);pointFromEvent(event)});canvas.addEventListener('pointermove',event=>{if(canvas.hasPointerCapture(event.pointerId))pointFromEvent(event)});

['note','velocity','temperature','morph'].forEach(id=>{$(`#${id}`).addEventListener('input',()=>{const value=+$('#'+id).value;$(`#${id}Value`).textContent=id==='note'?String(value):id==='morph'?`${value.toFixed(1)} s`:value.toFixed(2);scheduleControl();});});
$('#start').onclick=connectAndStart;
$('#noteOff').onclick=()=>{send({type:'note_off'});heldKeys=[];};
$('#reseed').onclick=()=>send({type:'reseed',seed:+$('#seed').value});
$('#stop').onclick=()=>{send({type:'stop'});heldKeys=[];setTimeout(()=>{ws?.close();resetAudio();},250);};

const keyboard={a:48,w:49,s:50,e:51,d:52,f:53,t:54,g:55,y:56,h:57,u:58,j:59,k:60};
window.addEventListener('keydown',event=>{const key=event.key.toLowerCase();if(event.repeat||!(key in keyboard)||event.target.matches('input'))return;heldKeys=heldKeys.filter(value=>value!==key);heldKeys.push(key);$('#note').value=keyboard[key];$('#noteValue').textContent=$('#note').value;if(!ws||ws.readyState!==WebSocket.OPEN)connectAndStart();else scheduleControl();});
window.addEventListener('keyup',event=>{const key=event.key.toLowerCase(),wasActive=heldKeys.at(-1)===key;heldKeys=heldKeys.filter(value=>value!==key);if(!wasActive)return;if(heldKeys.length){const fallbackKey=heldKeys.at(-1);$('#note').value=keyboard[fallbackKey];$('#noteValue').textContent=$('#note').value;scheduleControl();}else send({type:'note_off'});});

Promise.allSettled([loadEvaluation(),checkRuntime()]).then(results=>results.forEach(value=>{if(value.status==='rejected')console.error(value.reason)}));
setInterval(checkRuntime,30000);
