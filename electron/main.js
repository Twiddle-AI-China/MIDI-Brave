/**
 * Atlas Flow desktop shell.
 *
 * The app is a local inference server plus a web UI; this wraps both in a
 * window so there is no terminal and no browser tab to manage.
 *
 * Packaged, it is self-contained: `resources/atlas-flow-server/` holds a frozen
 * Python, torch, the 85 MB of weights, the atlas and the web assets, so there
 * is nothing to install and nothing to fetch. Unpackaged, it falls back to a
 * local PyInstaller build and then to the repo's own interpreter, so `npm
 * start` in a checkout still works without a packaging round first.
 *
 * Nothing here shells out. The previous version spawned bash on a .sh
 * launcher, which is the single reason this could not run on Windows.
 */
const {app, BrowserWindow, shell, Menu} = require('electron');
const {spawn, execFile} = require('node:child_process');
const {createWriteStream, existsSync, mkdirSync} = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const WINDOWS = process.platform === 'win32';
const EXE = WINDOWS ? 'atlas-flow-server.exe' : 'atlas-flow-server';

let server = null;
let window = null;

/** Desktop apps have no business asking for a click before making sound. */
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

/**
 * Where the repo is, when there is one.
 *
 * Packaged, __dirname lives inside the app bundle, so the path is recorded at
 * build time by prepack.js. A packaged app that ships its own server never
 * needs this; it is only the development fallbacks that do.
 */
function repoRoot() {
  if (process.env.ATLAS_REPO_ROOT) return process.env.ATLAS_REPO_ROOT;
  try {
    const recorded = require('./config.json').root;
    if (recorded && existsSync(recorded)) return recorded;
  } catch (_error) { /* a packaged build may legitimately have no config.json */ }
  return path.resolve(__dirname, '..');
}

/**
 * Decide how to start the server, in descending order of self-containment.
 *
 * Returns null when nothing usable is installed, so the window can say which
 * of the three it looked for rather than just failing.
 */
function launchPlan() {
  const bundled = path.join(process.resourcesPath || '', 'atlas-flow-server', EXE);
  if (existsSync(bundled)) {
    return {kind: 'bundled', command: bundled, args: [], cwd: path.dirname(bundled)};
  }
  const root = repoRoot();
  const built = path.join(root, 'build', 'pyi', 'atlas-flow-server', EXE);
  if (existsSync(built)) {
    return {kind: 'built', command: built, args: [], cwd: path.dirname(built)};
  }
  const python = process.env.ATLAS_LOCAL_PYTHON || path.join(
    root, '.venv-local', WINDOWS ? 'Scripts' : 'bin', WINDOWS ? 'python.exe' : 'python');
  if (existsSync(python)) {
    return {
      kind: 'source',
      command: python,
      args: ['-m', 'midibrave.atlas_flow_local'],
      cwd: root,
      env: {PYTHONPATH: path.join(root, 'src')},
    };
  }
  return null;
}

/**
 * Writable scratch for renders, takes and recordings.
 *
 * Not "cache": Electron keeps its own `Cache/` in this same folder, and macOS
 * filesystems are case-insensitive by default, so that name collided with it --
 * our takes landed inside the HTTP cache Electron is free to clear.
 */
function cacheDir() {
  return path.join(app.getPath('userData'), 'model-cache');
}

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

async function waitForServer(port, seconds = 240) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await healthy(port)) return true;
    if (server && server.exitCode !== null) return false;   // it died on us
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  return false;
}

function splash(message, detail) {
  const escape = text => String(text || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const page = `<!doctype html><meta charset="utf-8">
<style>
  html,body{height:100%;margin:0;background:#000;color:#f2f2f2;
    font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;display:grid;place-items:center}
  div{text-align:center;max-width:44em;padding:0 2em}
  b{font-size:15px;letter-spacing:.14em;display:block;margin-bottom:.8em}
  p{color:#8a8a8a;white-space:pre-wrap;text-align:left}
</style><div><b>ATLAS FLOW</b><p>${escape(message)}\n\n${escape(detail)}</p></div>`;
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

  const plan = launchPlan();
  if (!plan) {
    splash('Cannot start — no inference server was found.',
      'Looked for, in order:\n'
      + `  ${path.join(process.resourcesPath || '(unpackaged)', 'atlas-flow-server', EXE)}\n`
      + `  ${path.join(repoRoot(), 'build', 'pyi', 'atlas-flow-server', EXE)}\n`
      + `  ${path.join(repoRoot(), '.venv-local', WINDOWS ? 'Scripts' : 'bin', 'python')}\n\n`
      + 'A packaged build should carry the first of these. From a checkout, run\n'
      + '  npm --prefix electron run build:server');
    return;
  }

  const port = await freePort();
  splash('Loading the model…',
    plan.kind === 'bundled'
      ? 'First launch is the slow one: the model and its runtime are being read\n'
        + 'off disk for the first time. Later launches start in a few seconds.'
      : `Running from ${plan.kind === 'built' ? 'a local server build' : 'source'}.`);

  mkdirSync(cacheDir(), {recursive: true});
  server = spawn(plan.command, [...plan.args, '--port', String(port)], {
    cwd: plan.cwd,
    env: {
      ...process.env,
      ...(plan.env || {}),
      ATLAS_LOCAL_CACHE: cacheDir(),
      // If this shell is force-quit, before-quit never runs; the server watches
      // this pid and exits by itself rather than orphaning the port and the GPU.
      ATLAS_PARENT_PID: String(process.pid),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  // A packaged app has nowhere to print, so keep a log in the user data folder.
  // It is the only way to learn why a launch failed on someone else's machine.
  const log = [];
  let logFile = null;
  try {
    logFile = createWriteStream(path.join(app.getPath('userData'), 'desktop.log'), {flags: 'a'});
    logFile.write(`\n--- launch ${new Date().toISOString()} ---\n`
      + `${plan.kind}: ${plan.command}\nport ${port} on ${process.platform}/${os.arch()}\n`);
  } catch (_error) { /* logging is best effort */ }
  const remember = chunk => {
    const text = chunk.toString();
    log.push(text);
    if (log.length > 60) log.shift();
    if (logFile) logFile.write(text);
  };
  server.stdout.on('data', remember);
  server.stderr.on('data', remember);
  server.on('error', error => remember(`spawn failed: ${error.message}\n`));
  server.on('exit', (code, signal) => remember(`server exited code=${code} signal=${signal}\n`));

  if (await waitForServer(port)) {
    window.loadURL(`http://127.0.0.1:${port}`);
  } else {
    splash('The model server did not come up.',
      (log.join('') || '(it produced no output at all)').slice(-1600));
  }
}

/**
 * Stop the server.
 *
 * Windows has no SIGTERM: Node's kill() maps it onto TerminateProcess, which
 * leaves any child the server spawned running. taskkill /T walks the tree.
 */
function stopServer() {
  if (!server || server.exitCode !== null) return;
  if (WINDOWS) {
    execFile('taskkill', ['/pid', String(server.pid), '/T', '/F'], () => {});
    return;
  }
  server.kill('SIGTERM');
  setTimeout(() => server && server.exitCode === null && server.kill('SIGKILL'), 4000);
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{role: 'appMenu'}] : []),
    {role: 'fileMenu'},
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
