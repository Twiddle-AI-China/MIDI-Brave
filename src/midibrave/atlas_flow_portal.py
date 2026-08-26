from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Iterable

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web


_HTTP = web.AppKey("atlas_flow_portal_http", ClientSession)
_EVALUATION = web.AppKey("atlas_flow_evaluation_root", Path)
_WEB = web.AppKey("atlas_flow_web_root", Path)
_RUNTIME = web.AppKey("atlas_flow_runtime_url", str)


def _safe_file(root: Path, tail: str) -> Path:
    target = (root / tail).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise web.HTTPForbidden(text="invalid path")
    if not target.is_file():
        raise web.HTTPNotFound(text="file not found")
    return target


@web.middleware
async def _headers(request: web.Request, handler):
    response = await handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = (
        "no-store" if request.path.startswith("/api/") else "public, max-age=60"
    )
    return response


def create_app(
    evaluation_root: str | Path,
    web_root: str | Path,
    runtime_url: str,
) -> web.Application:
    evaluation = Path(evaluation_root).resolve()
    frontend = Path(web_root).resolve()
    app = web.Application(middlewares=[_headers], client_max_size=64 * 1024)
    app[_EVALUATION] = evaluation
    app[_WEB] = frontend
    app[_RUNTIME] = runtime_url.rstrip("/")

    async def startup(application: web.Application) -> None:
        application[_HTTP] = ClientSession(timeout=ClientTimeout(total=5.0))

    async def cleanup(application: web.Application) -> None:
        await application[_HTTP].close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    async def static(request: web.Request) -> web.FileResponse:
        filename = request.match_info.get("filename") or "index.html"
        return web.FileResponse(_safe_file(request.app[_WEB], filename))

    async def evaluation_report(request: web.Request) -> web.Response:
        path = request.app[_EVALUATION] / "evaluation.json"
        if not path.is_file():
            return web.json_response({
                "pipeline_state": "pending",
                "quality_state": "not_evaluated",
                "passed": False,
                "gates": {},
                "rows": [],
            }, status=202)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return web.json_response({
                "pipeline_state": "error", "passed": False,
                "message": str(error),
            }, status=503)
        payload["checkpoint"] = Path(str(payload.get("checkpoint", ""))).name
        return web.json_response(payload)

    async def evaluation_file(request: web.Request) -> web.FileResponse:
        return web.FileResponse(_safe_file(
            request.app[_EVALUATION], request.match_info["tail"],
        ))

    async def runtime_status(request: web.Request) -> web.Response:
        try:
            async with request.app[_HTTP].get(
                request.app[_RUNTIME] + "/api/runtime-status",
            ) as upstream:
                payload = await upstream.json(content_type=None)
                return web.json_response(payload, status=upstream.status)
        except (OSError, asyncio.TimeoutError, ValueError) as error:
            return web.json_response({
                "ok": False, "state": "offline", "message": str(error),
            }, status=503)

    async def runtime_ws(request: web.Request) -> web.WebSocketResponse:
        browser = web.WebSocketResponse(heartbeat=15.0, max_msg_size=64 * 1024)
        await browser.prepare(request)
        upstream = None
        try:
            upstream = await request.app[_HTTP].ws_connect(
                request.app[_RUNTIME] + "/runtime", heartbeat=15.0,
                max_msg_size=64 * 1024,
            )

            async def browser_to_runtime() -> None:
                async for message in browser:
                    if message.type == WSMsgType.TEXT:
                        await upstream.send_str(message.data)
                    elif message.type == WSMsgType.BINARY:
                        await upstream.send_bytes(message.data)
                    elif message.type in {WSMsgType.CLOSE, WSMsgType.ERROR}:
                        break

            async def runtime_to_browser() -> None:
                async for message in upstream:
                    if message.type == WSMsgType.TEXT:
                        await browser.send_str(message.data)
                    elif message.type == WSMsgType.BINARY:
                        await browser.send_bytes(message.data)
                    elif message.type in {WSMsgType.CLOSE, WSMsgType.ERROR}:
                        break

            forward = asyncio.create_task(browser_to_runtime())
            backward = asyncio.create_task(runtime_to_browser())
            done, pending = await asyncio.wait(
                {forward, backward}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        except (OSError, asyncio.TimeoutError, ValueError) as error:
            if not browser.closed:
                await browser.send_json({
                    "type": "error", "message": f"runtime offline: {error}",
                })
        finally:
            if upstream is not None:
                await upstream.close()
            if not browser.closed:
                await browser.close()
        return browser

    async def health(request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "evaluation": (request.app[_EVALUATION] / "evaluation.json").is_file(),
            "runtimeProxy": request.app[_RUNTIME],
        })

    app.router.add_get("/", static)
    app.router.add_get("/{filename:app\\.js|styles\\.css|pcm-player-worklet\\.js}", static)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/evaluation", evaluation_report)
    app.router.add_get("/api/runtime-status", runtime_status)
    app.router.add_get("/runtime", runtime_ws)
    app.router.add_get("/evaluation/{tail:.*}", evaluation_file)
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the Atlas Flow evaluation portal")
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8791")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    web.run_app(
        create_app(args.evaluation_root, args.web_root, args.runtime_url),
        host=args.host, port=args.port,
    )


if __name__ == "__main__":
    main()
