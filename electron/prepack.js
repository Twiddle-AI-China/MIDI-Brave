/**
 * Record where the repo lives, so the packaged app can find the interpreter and
 * the model. Until Python is bundled, the .app is a shell around this checkout
 * rather than a self-contained program, and this is the honest way to say so.
 */
const {writeFileSync} = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
writeFileSync(path.join(__dirname, 'config.json'), JSON.stringify({
  root,
  note: 'Written by prepack.js at build time. The app runs .venv-local and '
      + 'local-model from this checkout; bundling Python is the next step.',
}, null, 2) + '\n');
console.log(`packaged app will use repo root: ${root}`);
