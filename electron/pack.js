/**
 * Package the desktop app for whichever platform is running this.
 *
 * Cross-compiling is not attempted: the frozen server in extraResource is a
 * native binary with native torch libraries inside it, so a macOS host can only
 * produce the macOS app and a Windows host only the Windows one. Both are built
 * by CI, on their own runners.
 */
const {packager} = require('@electron/packager');
const {existsSync} = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const server = path.join(root, 'build', 'pyi', 'atlas-flow-server');

async function main() {
  if (!existsSync(server)) {
    console.error(`no frozen server at ${server} — run: npm run build:server`);
    process.exit(1);
  }
  const paths = await packager({
    dir: __dirname,
    name: 'AtlasFlow',
    out: path.join(__dirname, 'dist'),
    overwrite: true,
    appBundleId: 'ai.twiddle.atlasflow',
    appVersion: require('./package.json').version,
    // The whole point of v1: the server travels inside the app.
    extraResource: [server],
    // node_modules holds electron itself, which packager supplies separately.
    ignore: [/^\/node_modules/, /^\/dist/],
    // Unsigned builds are fine for internal distribution; macOS users get one
    // right-click-open, which the README explains.
    osxSign: false,
  });
  for (const built of paths) console.log(`packaged: ${built}`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
