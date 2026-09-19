/**
 * Freeze the inference server into build/pyi/atlas-flow-server/.
 *
 * Written in Node rather than as a shell script so one command works on macOS
 * and Windows alike -- the previous bash launcher was the only reason this
 * project could not be built on Windows.
 *
 * It picks an interpreter in this order: $ATLAS_LOCAL_PYTHON, the repo's
 * .venv-local, then whatever `python3`/`python` is on PATH. That interpreter
 * needs torch, soundfile, numpy, scipy, aiohttp and pyinstaller; see the README.
 */
const {spawnSync} = require('node:child_process');
const {existsSync} = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const windows = process.platform === 'win32';

function interpreter() {
  if (process.env.ATLAS_LOCAL_PYTHON) return process.env.ATLAS_LOCAL_PYTHON;
  const venv = path.join(root, '.venv-local',
    windows ? 'Scripts' : 'bin', windows ? 'python.exe' : 'python');
  if (existsSync(venv)) return venv;
  return windows ? 'python' : 'python3';
}

const python = interpreter();
const check = spawnSync(python, ['-c', 'import PyInstaller, torch, soundfile, aiohttp'],
  {stdio: 'pipe', encoding: 'utf8'});
if (check.status !== 0) {
  console.error(`\n${python} cannot import what the build needs:\n`
    + `${(check.stderr || check.error?.message || '').trim()}\n\n`
    + 'Install them with:\n'
    + `  ${python} -m pip install torch soundfile numpy scipy aiohttp pyinstaller\n`);
  process.exit(1);
}

console.log(`freezing the server with ${python}`);
const build = spawnSync(python, [
  '-m', 'PyInstaller',
  path.join(root, 'packaging', 'atlas-flow-server.spec'),
  '--noconfirm',
  '--distpath', path.join(root, 'build', 'pyi'),
  '--workpath', path.join(root, 'build', 'pyi-work'),
  '--log-level', 'WARN',
], {cwd: root, stdio: 'inherit', env: {...process.env, ATLAS_PACK_ROOT: root}});

if (build.status !== 0) process.exit(build.status ?? 1);

const exe = path.join(root, 'build', 'pyi', 'atlas-flow-server',
  windows ? 'atlas-flow-server.exe' : 'atlas-flow-server');
if (!existsSync(exe)) {
  console.error(`PyInstaller reported success but ${exe} is not there`);
  process.exit(1);
}
console.log(`built ${exe}`);
