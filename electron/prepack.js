/**
 * Check that the frozen server exists before packaging, and record the repo
 * path for development fallbacks.
 *
 * A packaged app carries `resources/atlas-flow-server/` and never touches the
 * checkout. config.json is only read when that folder is absent -- an
 * unpackaged `npm start`, or a deliberately server-less build -- and is what
 * lets those fall back to a local build or the repo's interpreter.
 */
const {existsSync, writeFileSync} = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const exe = process.platform === 'win32' ? 'atlas-flow-server.exe' : 'atlas-flow-server';
const server = path.join(root, 'build', 'pyi', 'atlas-flow-server');

if (!existsSync(path.join(server, exe))) {
  console.error(
    `\nNo frozen server at ${path.join(server, exe)}.\n\n`
    + 'Packaging without it produces an app that falls back to a checkout,\n'
    + 'which is the thing this build exists to stop doing. Build it first:\n\n'
    + '  npm --prefix electron run build:server\n');
  process.exit(1);
}

writeFileSync(path.join(__dirname, 'config.json'), JSON.stringify({
  root,
  note: 'Written by prepack.js at build time. Only used when the packaged app '
      + 'has no bundled server, i.e. running from a checkout.',
}, null, 2) + '\n');
console.log(`bundling server from ${server}`);
