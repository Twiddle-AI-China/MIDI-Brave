# PyInstaller spec for the Atlas Flow inference server.
#
#   pyinstaller packaging/atlas-flow-server.spec --noconfirm --distpath build/pyi
#
# Produces build/pyi/atlas-flow-server/ -- an interpreter, torch, the model
# weights, the atlas and the web UI, with no dependency on a checkout or a
# virtualenv. The Electron shell ships this folder in its resources and spawns
# the binary inside it.
#
# Onedir, deliberately, not onefile: onefile unpacks several hundred megabytes
# of torch to a temporary directory on every launch, which turns a cold start
# into tens of seconds and leaves debris behind if the app is force-quit.
#
# Everything the bundle needs at runtime is collected as data, so paths inside
# it are relative to sys._MEIPASS and identical on macOS and Windows.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(os.environ.get("ATLAS_PACK_ROOT", Path.cwd())).resolve()
MODEL = Path(os.environ.get("ATLAS_PACK_MODEL", ROOT / "local-model")).resolve()

for required in (ROOT / "src" / "midibrave" / "atlas_flow_local.py",
                 ROOT / "atlas-flow-web-demo" / "index.html",
                 ROOT / "configs" / "atlas_flow" / "kraken_pad_v1.yaml",
                 MODEL / "atlas-flow-pad-v1-weights.pt",
                 MODEL / "pad-top50-atlas.npz"):
    if not required.exists():
        raise SystemExit(f"packaging needs {required}, which is not there")

datas = [
    (str(ROOT / "atlas-flow-web-demo"), "atlas-flow-web-demo"),
    (str(ROOT / "configs" / "atlas_flow" / "kraken_pad_v1.yaml"), "configs/atlas_flow"),
    (str(MODEL / "atlas-flow-pad-v1-weights.pt"), "local-model"),
    (str(MODEL / "pad-top50-atlas.npz"), "local-model"),
]
# The gate readout and the held-out audition are nice to have and add 36 MB;
# include them when they are present, skip them when they are not.
evaluation = MODEL / "evaluation"
if evaluation.is_dir():
    datas.append((str(evaluation), "local-model/evaluation"))

# soundfile carries libsndfile as package data, and it is loaded by ctypes at
# import time, so PyInstaller's import graph never sees it.
datas += collect_data_files("soundfile")
binaries = collect_dynamic_libs("soundfile")

hiddenimports = [
    "midibrave.atlas_flow_demo_server",
    "midibrave.atlas_flow_runtime",
    "midibrave.atlas_flow_runtime_server",
    "midibrave.atlas_flow_stream",
    # aiohttp resolves these lazily; without them the server imports and then
    # fails on the first request instead of failing at build time.
    "aiohttp",
    "scipy.signal",
    "scipy.special",
]

# torchaudio, torchvision and the notebook stack are not used and are large.
# torch's own submodules are deliberately NOT excluded: torch.utils.data
# imports torch.distributed unconditionally, so dropping it produced a bundle
# that built cleanly and then died on the first import.
excludes = [
    "torchaudio", "torchvision", "tkinter", "matplotlib", "IPython",
    "pytest", "notebook", "jupyter",
]

analysis = Analysis(
    [str(ROOT / "packaging" / "server_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="atlas-flow-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX mangles the torch dylibs and produces a binary that will not load.
    upx=False,
    console=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="atlas-flow-server",
)
