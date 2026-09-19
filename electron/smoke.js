/**
 * Check that the *packaged* app is actually self-contained.
 *
 * packaging/smoke_test.py exercises the frozen server in depth, including the
 * websocket. This asks a narrower question that only a packaged build can
 * answer: did the server end up inside the app, and does it run from in there
 * with no checkout, no virtualenv and no PYTHONPATH?
 *
 * Pure Node on purpose -- if this needed Python to run, it could not tell the
 * difference between a self-contained app and one leaning on the machine.
 *
 *   node electron/smoke.js                     # finds dist/ for this platform
 *   node electron/smoke.js --app <path>        # or point it at one
 */
const {spawn} = require('node:child_process');
const {existsSync, readdirSync} = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

const WINDOWS = process.platform === 'win32';
const EXE = WINDOWS ? 'atlas-flow-server.exe' : 'atlas-flow-server';
const failures = [];

function check(name, ok, detail = '') {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  -- ${detail}` : ''}`);
  if (!ok) failures.push(name);
  return ok;
}

/** The packaged app's resources folder, wherever this platform put it. */
function resourcesDir(explicit) {
  const dist = explicit || path.join(__dirname, 'dist');
  if (!existsSync(dist)) return null;
  for (const entry of readdirSync(dist)) {
    const base = path.join(dist, entry);
    const mac = path.join(base, 'AtlasFlow.app', 'Contents', 'Resources');
    if (existsSync(mac)) return mac;
    const other = path.join(base, 'resources');
    if (existsSync(other)) return other;
  }
  // Or the caller pointed us straight at an .app / app folder.
  const direct = path.join(dist, 'Contents', 'Resources');
  if (existsSync(direct)) return direct;
  const plain = path.join(dist, 'resources');
  return existsSync(plain) ? plain : null;
}

const freePort = () => new Promise((resolve, reject) => {
  const probe = net.createServer();
  probe.unref();
  probe.on('error', reject);
  probe.listen(0, '127.0.0.1', () => {
    const {port} = probe.address();
    probe.close(() => resolve(port));
  });
});

const fetchText = (port, route, timeout = 20000) => new Promise(resolve => {
  const request = http.get({host: '127.0.0.1', port, path: route, timeout}, response => {
    const chunks = [];
    response.on('data', chunk => chunks.push(chunk));
    response.on('end', () => resolve({status: response.statusCode, body: Buffer.concat(chunks)}));
  });
  request.on('error', () => resolve({status: 0, body: Buffer.alloc(0)}));
  request.on('timeout', () => { request.destroy(); resolve({status: 0, body: Buffer.alloc(0)}); });
});

function postJson(port, route, payload, timeout = 600000) {
  const body = Buffer.from(JSON.stringify(payload));
  return new Promise(resolve => {
    const request = http.request({
      host: '127.0.0.1', port, path: route, method: 'POST', timeout,
      headers: {'content-type': 'application/json', 'content-length': body.length},
    }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => {
        try {
          resolve({status: response.statusCode, json: JSON.parse(Buffer.concat(chunks))});
        } catch (_error) {
          resolve({status: response.statusCode, json: {}});
        }
      });
    });
    request.on('error', error => resolve({status: 0, json: {error: error.message}}));
    request.on('timeout', () => { request.destroy(); resolve({status: 0, json: {}}); });
    request.end(body);
  });
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  const flag = process.argv.indexOf('--app');
  const resources = resourcesDir(flag > -1 ? process.argv[flag + 1] : null);
  if (!check('the packaged app was found', !!resources, resources || 'no dist/ build')) {
    return 1;
  }
  const server = path.join(resources, 'atlas-flow-server', EXE);
  if (!check('the server travels inside the app', existsSync(server), server)) return 1;

  const internal = path.join(resources, 'atlas-flow-server', '_internal');
  check('the weights travel with it',
    existsSync(path.join(internal, 'local-model', 'atlas-flow-pad-v1-weights.pt')));
  check('the web UI travels with it',
    existsSync(path.join(internal, 'atlas-flow-web-demo', 'index.html')));

  const port = await freePort();
  console.log(`  starting ${server} on ${port}`);
  // Deliberately hostile environment: no PYTHONPATH, no venv on PATH, cwd
  // somewhere unrelated. If the app needs the checkout, it fails here.
  const environment = {...process.env, ATLAS_LOCAL_CACHE: path.join(__dirname, 'dist', 'smoke-cache')};
  delete environment.PYTHONPATH;
  delete environment.ATLAS_LOCAL_MODEL;
  delete environment.ATLAS_REPO_ROOT;
  const child = spawn(server, ['--port', String(port)], {
    cwd: path.parse(process.cwd()).root,
    env: environment,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });

  try {
    const deadline = Date.now() + 420000;
    let ready = false;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) break;
      if ((await fetchText(port, '/api/health', 3000)).status === 200) { ready = true; break; }
      await sleep(1000);
    }
    if (!check('it serves from inside the bundle', ready, `exit ${child.exitCode}`)) {
      console.log(output.slice(-2000));
      return 1;
    }

    const status = await fetchText(port, '/api/status');
    let parsed = {};
    try { parsed = JSON.parse(status.body); } catch (_error) { /* reported below */ }
    check('the atlas loaded', (parsed.points || []).length === 50,
      `${(parsed.points || []).length} presets, device ${parsed.cuda}`);

    const index = await fetchText(port, '/');
    check('the UI is served', index.status === 200 && index.body.includes('ATLAS FLOW'),
      `${index.body.length} bytes`);

    const render = await postJson(port, '/api/render', {
      steps: [{pca: [0.1, 0.2, 0, 0, 0, 0, 0, 0], notes: [50], seconds: 2.0}],
      seed: 20260822, velocity: 0.8,
    });
    check('it renders audio', render.status === 200 && render.json.bytes > 4000,
      render.status === 200
        ? `${render.json.bytes} bytes, peak ${render.json.peakDbfs} dBFS`
        : JSON.stringify(render.json).slice(0, 160));
    check('the audio is not silence',
      render.status === 200 && Number(render.json.peakDbfs) > -60);
  } finally {
    if (child.exitCode === null) {
      if (WINDOWS) spawn('taskkill', ['/pid', String(child.pid), '/T', '/F']);
      else child.kill('SIGTERM');
      await sleep(3000);
    }
  }

  if (failures.length) {
    console.log(`\n${failures.length} check(s) failed: ${failures.join(', ')}`);
    return 1;
  }
  console.log('\npackaged app is self-contained');
  return 0;
}

main().then(code => process.exit(code)).catch(error => {
  console.error(error);
  process.exit(1);
});
