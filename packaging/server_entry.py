"""Entry point for the frozen Atlas Flow server.

PyInstaller needs a real script to analyse, and the frozen binary needs the
model and web assets found inside the bundle rather than next to a checkout.
Both are handled here so ``atlas_flow_local`` stays a plain module that also
runs from source.

multiprocessing.freeze_support() must be the first thing that happens: torch's
DataLoader and a few of its internals spawn helpers, and on Windows every one
of those re-executes this binary. Without the guard each helper would start a
second web server and the app would deadlock on the port.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def main() -> int:
    multiprocessing.freeze_support()

    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    # Let the module's own resolution find the payload we shipped beside it.
    os.environ.setdefault("ATLAS_LOCAL_MODEL", str(bundle / "local-model"))
    # The cache has to be writable, and a bundle usually is not: on macOS it
    # sits inside a signed .app, and under Program Files it is read-only. Put
    # it where the platform says user data goes.
    if not os.environ.get("ATLAS_LOCAL_CACHE"):
        os.environ["ATLAS_LOCAL_CACHE"] = str(user_cache())

    from midibrave.atlas_flow_local import main as run
    return run()


def user_cache() -> Path:
    """Per-user writable scratch, following each platform's convention.

    Called "model-cache" rather than "cache" because the desktop shell keeps
    this beside Electron's own `Cache/`, and a case-insensitive filesystem
    treats those as the same folder.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "AtlasFlow" / "model-cache"


if __name__ == "__main__":
    raise SystemExit(main())
