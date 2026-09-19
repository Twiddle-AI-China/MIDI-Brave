/**
 * Atlas Flow desktop shell.
 *
 * The app is a local Python server plus a web UI; this wraps both in a window so
 * there is no terminal and no browser tab to manage. It deliberately does NOT
 * bundle Python or torch yet — it runs the interpreter in the repo's .venv-local
 * (850 MB of it is torch, against 85 MB of model), which is the next step.
 */
const {app, BrowserWindow, dialog, shell, Menu} = require('electron');
const {spawn} = require('node:child_process');
const {createWriteStream, existsSync, mkdirSync} = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

// Packaged, __dirname lives inside the .app, so the repo path is recorded at
// build time; unpackaged, the repo is simply the parent directory.
function repoRoot() {
  if (process.env.ATLAS_REPO_ROOT) return process.env.ATLAS_REPO_ROOT;
  try {
    const recorded = require('./config.json').root;
    if (recorded && existsSync(recorded)) return recorded;
  } catch (_error) { /* unpackaged build has no config.json */ }
  return path.resolve(__dirname, '..');
}

const ROOT = repoRoot();
const LAUNCHER = path.join(ROOT, 'scripts', 'atlas_flow', 'local_demo.sh');
const PYTHON = process.env.ATLAS_LOCAL_PYTHON || path.join(ROOT, '.venv-local', 'bin', 'python');
const MODEL = process.env.ATLAS_LOCAL_MODEL || path.join(ROOT, 'local-model');

let server = null;
let window = null;

/** Desktop apps have no business asking for a click before making sound. */
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.unref();
    probe.on('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const {port} = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

function healthy(port) {
  return new Promise(resolve => {
    const request = http.get(
      {host: '127.0.0.1', port, path: '/api/health', timeout: 2000},
      response => {
        response.resume();
        resolve(response.statusCode === 200);
      },
    );
    request.on('error', () => resolve(false));
    request.on('timeout', () => { request.destroy(); resolve(false); });
  });
}

async function waitForServer(port, seconds = 180) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await healthy(port)) return true;
    if (server && server.exitCode !== null) return false;   // it died on us
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  return false;
}

function missingPieces() {
  const missing = [];
  if (!existsSync(PYTHON)) missing.push(`interpreter: ${PYTHON}`);
  if (!existsSync(path.join(MODEL, 'atlas-flow-pad-v1-weights.pt'))) {
    missing.push(`weights: ${path.join(MODEL, 'atlas-flow-pad-v1-weights.pt')}`);
  }
  if (!existsSync(path.join(MODEL, 'pad-top50-atlas.npz'))) {
    missing.push(`atlas: ${path.join(MODEL, 'pad-top50-atlas.npz')}`);
  }
  return missing;
}

function splash(message, detail) {
  const page = `<!doctype html><meta charset="utf-8">
<style>
  html,body{height:100%;margin:0;background:#000;color:#f2f2f2;
    font:13px/1.6 ui-monospace,Menlo,monospace;display:grid;place-items:center}
  div{text-align:center;max-width:40em;padding:0 2em}
  b{font-size:15px;letter-spacing:.14em;display:block;margin-bottom:.8em}
  p{color:#8a8a8a;white-space:pre-wrap}
</style><div><b>ATLAS FLOW</b><p>${message}\n\n${detail || ''}</p></div>`;
  window.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(page));
}

async function start() {
  window = new BrowserWindow({
    width: 1560,
    height: 900,
    minWidth: 860,
    minHeight: 560,
    backgroundColor: '#000000',
    title: 'Atlas Flow',
    webPreferences: {nodeIntegration: false, contextIsolation: true},
  });
  // External links belong in the user's browser, not in this window.
  window.webContents.setWindowOpenHandler(({url}) => {
    shell.openExternal(url);
    return {action: 'deny'};
  });

  const missing = missingPieces();
  if (missing.length) {
    splash('Cannot start — something is missing.',
      missing.join('\n') + '\n\nRun scripts/atlas_flow/fetch_local_model.sh first.');
    return;
  }

  const port = await freePort();
  splash('Loading the model…', 'First run compiles Metal kernels, which takes a moment.');

  server = spawn('bash', [LAUNCHER], {
    cwd: ROOT,
    // If this shell is force-quit, before-quit never runs; the server watches
    // this pid and exits by itself rather than orphaning a GPU process.
    env: {...process.env, ATLAS_LOCAL_PORT: String(port),
          ATLAS_PARENT_PID: String(process.pid)},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  // A packaged app has nowhere to print, so keep a log next to the model. It is
  // the only way to learn why a launch failed on someone else's machine.
  const log = [];
  let logFile = null;
  try {
    mkdirSync(path.join(MODEL, 'cache'), {recursive: true});
    logFile = createWriteStream(path.join(MODEL, 'cache', 'desktop.log'), {flags: 'a'});
    logFile.write(`\n--- launch ${new Date().toISOString()} port ${port} ---\n`);
  } catch (_error) { /* logging is best effort */ }
  const remember = chunk => {
    const text = chunk.toString();
    log.push(text);
    if (log.length > 40) log.shift();
    if (logFile) logFile.write(text);
  };
  server.stdout.on('data', remember);
  server.stderr.on('data', remember);
  server.on('error', error => remember(`spawn failed: ${error.message}\n`));
  server.on('exit', (code, signal) => remember(`server exited code=${code} signal=${signal}\n`));

  if (await waitForServer(port)) {
    window.loadURL(`http://127.0.0.1:${port}`);
  } else {
    splash('The model server did not come up.', log.join('').slice(-1200));
  }
}

function stopServer() {
  if (!server || server.exitCode !== null) return;
  server.kill('SIGTERM');
  setTimeout(() => server && server.exitCode === null && server.kill('SIGKILL'), 4000);
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {role: 'appMenu'},
    {role: 'editMenu'},
    {label: 'View', submenu: [
      {role: 'reload'}, {role: 'forceReload'}, {role: 'toggleDevTools'},
      {type: 'separator'}, {role: 'resetZoom'}, {role: 'zoomIn'}, {role: 'zoomOut'},
      {type: 'separator'}, {role: 'togglefullscreen'},
    ]},
    {role: 'windowMenu'},
  ]));
  start();
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && start());
});

app.on('window-all-closed', () => {
  stopServer();
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', stopServer);
process.on('exit', stopServer);
