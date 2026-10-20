"""Public website for the voice bot.

The browser talks only to this app. Connecting starts a sticky session on
VoiceBot and stores the session token in a cookie. Offer and ICE-candidate
requests are forwarded with that token, which a browser preflight would not
send to the bot itself.
"""

import asyncio
import time
from pathlib import Path

import modal

CLIENT_ROOT = Path("/client")
BOT_APP_NAME = "phonon-pipecat-bot"
BOT_SERVER_NAME = "VoiceBot"
SESSION_COOKIE = "bot_session"
SESSION_IDLE_SECONDS = 600


def _browser_ice_servers(specs: list) -> list[dict]:
    servers = []
    for spec in specs:
        if isinstance(spec, str):
            servers.append({"urls": spec})
        else:
            servers.append(spec)
    return servers


def _bot() -> modal.Server:
    return modal.Server.from_name(BOT_APP_NAME, BOT_SERVER_NAME)


async def _wait_healthy(url: str, token: str) -> dict:
    import httpx

    headers = {"Modal-Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + 180
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                response = await client.get(f"{url}/health", headers=headers)
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("bot did not become healthy")
            await asyncio.sleep(2)


def _client_api():
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    def token_from(request: Request) -> str | None:
        return request.cookies.get(SESSION_COOKIE)

    @api.post("/api/session")
    async def start_session(request: Request, response: Response):
        server = _bot()
        previous = token_from(request)
        if previous:
            try:
                await server.sessions.terminate.aio(previous)
            except Exception:
                pass
        session = await server.sessions.start.aio(idle_timeout=SESSION_IDLE_SECONDS)
        url = await server.get_url.aio()
        try:
            if not url:
                raise RuntimeError("VoiceBot has no URL")
            health = await _wait_healthy(url, session.token)
        except Exception:
            await server.sessions.terminate.aio(session.token)
            raise
        response.set_cookie(
            SESSION_COOKIE,
            session.token,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/",
            max_age=SESSION_IDLE_SECONDS * 2,
        )
        servers = _browser_ice_servers(health.get("ice_servers", []))
        return {"iceServers": servers, "iceConfig": {"iceServers": servers}}

    @api.delete("/api/session")
    async def end_session(request: Request):
        token = token_from(request)
        if token:
            await _bot().sessions.terminate.aio(token)
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @api.get("/api/keepalive")
    async def keepalive(request: Request):
        token = token_from(request)
        url = await _bot().get_url.aio()
        if not token or not url:
            return Response(status_code=401)
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.get(
                f"{url}/health",
                headers={"Modal-Authorization": f"Bearer {token}"},
            )
        if upstream.status_code != 200:
            return Response(status_code=upstream.status_code)
        return Response(status_code=204)

    @api.api_route("/api/offer", methods=["POST", "PATCH"])
    async def offer(request: Request):
        token = token_from(request)
        url = await _bot().get_url.aio()
        if not token or not url:
            return JSONResponse({"detail": "no session"}, status_code=401)
        body = await request.body()
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await client.request(
                request.method,
                f"{url}/api/offer",
                content=body,
                headers={
                    "Content-Type": request.headers.get("content-type", "application/json"),
                    "Modal-Authorization": f"Bearer {token}",
                },
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    api.mount("/", StaticFiles(directory=CLIENT_ROOT, html=True), name="client")
    return api
