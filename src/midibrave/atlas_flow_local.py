"""Start the Atlas Flow demo against a local model, on any platform.

This is what ``scripts/atlas_flow/local_demo.sh`` used to do in bash, moved into
Python so that Windows can run it too and so that the packaged desktop app has a
single entry point to freeze. The shell script now calls this module, and the
Electron shell calls either this module or the frozen binary built from it.

Everything it does is setup: find the model, pick a device, write a config whose
paths point at this machine, then hand over to the demo server's own ``main``.

Run it directly with::

    python -m midibrave.atlas_flow_local --port 18796

It needs two files, both produced by ``scripts/atlas_flow/fetch_local_model.sh``:

    <model>/atlas-flow-pad-v1-weights.pt   inference weights, 85 MiB
    <model>/pad-top50-atlas.npz            the legal-region atlas, 24 KiB

and optionally ``<model>/evaluation/`` for the gate readout in the settings
panel.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

WEIGHTS_NAME = "atlas-flow-pad-v1-weights.pt"
ATLAS_NAME = "pad-top50-atlas.npz"

# Only these keys are rewritten. The rest of the training config is left alone:
# the runtime reads the model section verbatim and must not be second-guessed
# here, and the remaining data paths are required fields it never opens.
REWRITTEN = ("atlas_path", "manifest", "audio_root", "feature_cache",
             "trajectory_cache", "output_root")


def frozen() -> bool:
    """True when running from a PyInstaller bundle rather than a checkout."""
    return getattr(sys, "frozen", False)


def bundle_root() -> Path:
    """Where the payload lives: the bundle when frozen, the repo when not."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def usable(device: str) -> bool:
    """Can this backend actually hold a tensor, not merely claim to exist?

    is_available() answers "is the framework present", which is not the same
    question. A macOS VM -- a CI runner, a remote session, a virtualised Mac --
    reports Metal as available and then fails the first allocation with an
    out-of-memory error against a pool that has nothing in it. Asking the
    backend to hold 256 floats separates the two cases in about a millisecond.
    """
    import torch
    try:
        probe = torch.zeros(256, device=device)
        return float((probe + 1.0).sum()) == 256.0
    except (RuntimeError, AssertionError):
        return False


def pick_device(requested: str | None) -> str:
    """Prefer the fastest backend that works, and settle for one that does.

    Metal roughly halves the plan on an M2 (283 ms against 567 ms, nearly all of
    it in the decoder) and CUDA is faster still, so try both before settling for
    CPU. An explicit --device always wins, including for reproducing a bug.
    """
    if requested:
        return requested
    try:
        import torch
    except ImportError:       # the caller will fail more usefully than we can
        return "cpu"
    if torch.cuda.is_available() and usable("cuda:0"):
        return "cuda:0"
    if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            and usable("mps")):
        return "mps"
    return "cpu"


def write_config(template: Path, destination: Path, values: dict[str, Path]) -> Path:
    """Copy the training config with this machine's paths substituted in.

    Done with a line rewrite rather than a YAML round-trip so the file a user
    opens still looks like the one in the repo, comments and ordering intact.
    """
    lines = template.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        match = re.match(r"^(\s+)(\w+):", line)
        if match and match.group(2) in values:
            out.append(f"{match.group(1)}{match.group(2)}: {values[match.group(2)].as_posix()}")
        else:
            out.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n", encoding="utf-8")
    return destination


def resolve(args: argparse.Namespace) -> list[str]:
    """Build the demo server's argv, or exit with something actionable."""
    root = bundle_root()
    model = args.model or Path(os.environ.get("ATLAS_LOCAL_MODEL") or root / "local-model")
    cache = args.cache or Path(os.environ.get("ATLAS_LOCAL_CACHE") or model / "cache")
    web = args.web_root or root / "atlas-flow-web-demo"
    template = args.config or root / "configs" / "atlas_flow" / "kraken_pad_v1.yaml"

    weights = model / WEIGHTS_NAME
    atlas = model / ATLAS_NAME
    missing = [str(path) for path in (weights, atlas, web, template) if not path.exists()]
    if missing:
        raise SystemExit(
            "atlas flow cannot start, these are missing:\n  "
            + "\n  ".join(missing)
            + "\n\nFetch the model with:  bash scripts/atlas_flow/fetch_local_model.sh"
        )

    for folder in ("takes", "audition"):
        (cache / folder).mkdir(parents=True, exist_ok=True)

    config = write_config(template, cache / "local_pad_v1.yaml", {
        "atlas_path": atlas,
        "manifest": model / "unused-manifest.jsonl",
        "audio_root": model / "unused-audio",
        "feature_cache": cache / "features",
        "trajectory_cache": cache / "trajectories",
        "output_root": cache / "runs",
    })

    evaluation = model / "evaluation"
    if not evaluation.is_dir():
        evaluation = cache / "no-evaluation"

    device = pick_device(args.device or os.environ.get("ATLAS_LOCAL_DEVICE"))
    port = args.port or int(os.environ.get("ATLAS_LOCAL_PORT") or 18796)
    parent = args.parent_pid or int(os.environ.get("ATLAS_PARENT_PID") or 0)
    threads = args.threads if args.threads is not None \
        else int(os.environ.get("ATLAS_LOCAL_THREADS") or 0)

    print(f"atlas flow, local: device={device}  threads={threads or 'auto'}  "
          f"http://{args.host}:{port}", flush=True)

    argv = [
        "--config", str(config),
        "--checkpoint", str(weights),
        "--cache-root", str(cache),
        "--evaluation-root", str(evaluation),
        "--web-root", str(web),
        "--profile", "local",
        "--device", device,
        "--threads", str(threads),
        "--host", args.host,
        "--port", str(port),
    ]
    if parent:
        argv += ["--parent-pid", str(parent)]
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Atlas Flow on this machine.")
    parser.add_argument("--model", type=Path, help="folder holding the weights and atlas")
    parser.add_argument("--cache", type=Path, help="writable scratch folder")
    parser.add_argument("--web-root", type=Path, help="folder holding index.html")
    parser.add_argument("--config", type=Path, help="training config to adapt")
    parser.add_argument("--device", help="cpu, mps, cuda:0 — default is the fastest present")
    parser.add_argument("--threads", type=int, help="torch CPU threads, 0 for auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--parent-pid", type=int,
                        help="exit when this process does, so a force-quit orphans nothing")
    parser.add_argument("--print-argv", action="store_true",
                        help="resolve everything and print it, without starting the server")
    return parser


def main(raw: list[str] | None = None) -> int:
    args = build_parser().parse_args(raw)
    argv = resolve(args)
    if args.print_argv:
        print("\n".join(argv))
        return 0
    from midibrave.atlas_flow_demo_server import main as serve
    return serve(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
