"""Local UI authentication, browser entry URLs, and authenticated discovery.

Keep this outside the app builders: only listening servers need a port-bound
guard. The ASGI wrapper forwards streaming responses without buffering them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import time
import urllib.error
import urllib.request
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import uvicorn
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

TOKEN_PATH = Path.home() / ".config" / "meeting-copilot" / "ui-token"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
PING_PATH = "/api/ui-ping"
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}")


def load_token(path: Path | None = None, *, _recreated: bool = False) -> str:
    """Read the shared user secret, creating it exclusively with private modes.

    A competing creator may have opened the file but not written it yet. Only
    that empty-file case is retried; all other read/write failures stop startup.
    """
    path = Path(path) if path is not None else TOKEN_PATH
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.stat().st_uid != os.getuid():
        raise PermissionError("UI token directory must belong to this user")
    if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        path.parent.chmod(0o700)
    flags = os.O_RDWR | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    else:
        with os.fdopen(fd, "w", encoding="ascii") as output:
            os.fchmod(output.fileno(), 0o600)
            token = secrets.token_urlsafe(32)
            output.write(token + "\n")
            output.flush()
            os.fsync(output.fileno())
        return token
    with os.fdopen(fd, "r", encoding="ascii") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise PermissionError("UI token must be a regular file owned by this user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            os.fchmod(source.fileno(), 0o600)
        for attempt in range(51):
            source.seek(0)
            token = source.read(258).strip()
            if token or attempt == 50:
                break
            time.sleep(0.01)
    if not token and not _recreated and time.time() - info.st_mtime > 2:
        # ★作る途中で落ちて空のまま残ったファイル。放っておくと以後ずっと画面が立たないので作り直す
        path.unlink()
        return load_token(path, _recreated=True)
    if not _TOKEN.fullmatch(token):
        raise ValueError("Invalid UI token file")
    return token


def require_loopback(host: str) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("ローカル UI は 127.0.0.1 / localhost / ::1 でのみ起動できます")


def browser_url(url: str) -> str:
    """Add the shared secret to a local browser entry URL, retaining its query."""
    parts = urlsplit(url)
    require_loopback(parts.hostname or "")
    if parts.scheme != "http" or parts.username is not None or parts.password is not None:
        raise ValueError("Expected a local HTTP UI URL")
    netloc = parts.netloc
    if parts.hostname == "::1":
        netloc = f"localhost:{parts.port}" if parts.port else "localhost"
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "token"]
    query.append(("token", load_token()))
    return urlunsplit(parts._replace(netloc=netloc, query=urlencode(query)))


def _proof(token: str, nonce: str, *, port: int, kind: str) -> str:
    message = f"meeting-copilot-ui:{kind}:{port}:{nonce}"
    return hmac.new(token.encode("ascii"), message.encode("ascii"),
                    hashlib.sha256).hexdigest()


def _matches(candidate: str, token: str) -> bool:
    return secrets.compare_digest(candidate.encode("utf-8"), token.encode("ascii"))


class _Guard:
    def __init__(self, app, *, port: int, kind: str, token: str) -> None:
        self.app, self.port, self.kind, self.token = app, port, kind, token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        headers: dict[str, list[str]] = {}
        for key, value in scope.get("headers", []):
            headers.setdefault(key.decode("latin-1").lower(), []).append(value.decode("latin-1"))
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        host = headers.get("host", [])
        origins = headers.get("origin", [])
        forbidden = (len(host) != 1 or host[0] not in hosts
                     or any(origin not in {f"http://{item}" for item in hosts} for origin in origins)
                     or len(origins) > 1
                     or any(site.lower() == "cross-site" for site in headers.get("sec-fetch-site", [])))
        if forbidden:
            await JSONResponse({"detail": "Forbidden"}, status_code=403)(scope, receive, send)
            return

        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        method = scope.get("method")
        response = None
        if method == "GET" and scope["path"] == PING_PATH:
            nonces = [value for key, value in query if key == "nonce"]
            if len(nonces) != 1 or not re.fullmatch(r"[a-fA-F0-9]{1,256}", nonces[0]):
                response = JSONResponse({"detail": "Expected a hex nonce"}, status_code=400)
            else:
                response = JSONResponse({"app": "meeting-copilot", "kind": self.kind,
                                         "proof": _proof(self.token, nonces[0], port=self.port, kind=self.kind)})
        else:
            entries = [value for key, value in query if key == "token"]
            if method == "GET" and entries:
                if len(entries) == 1 and _matches(entries[0], self.token):
                    remaining = urlencode([(key, value) for key, value in query if key != "token"])
                    path = scope.get("raw_path", scope["path"].encode("utf-8")).decode("latin-1")
                    location = f"http://{host[0]}{path}" + (f"?{remaining}" if remaining else "")
                    response = RedirectResponse(location, status_code=303)
                    response.set_cookie("mc_ui", self.token, httponly=True, samesite="strict",
                                        path="/", max_age=2592000)
                else:
                    response = self._unauthorized(method, headers)
            else:
                cookie = SimpleCookie()
                try:
                    cookie.load("; ".join(headers.get("cookie", [])))
                except Exception:
                    cookie.clear()
                candidates = headers.get("x-meeting-token", [])
                if "mc_ui" in cookie:
                    candidates = [*candidates, cookie["mc_ui"].value]
                if not any(_matches(value, self.token) for value in candidates):
                    response = self._unauthorized(method, headers)
        if response is not None:
            response.headers["cache-control"] = "no-store"
            response.headers["referrer-policy"] = "no-referrer"
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    @staticmethod
    def _unauthorized(method, headers):
        if method == "GET" and any("text/html" in value for value in headers.get("accept", [])):
            return HTMLResponse('<!doctype html><html lang="ja"><meta charset="utf-8">'
                                '<p>メニューバーか会議アシスタント.command から開き直してください</p></html>',
                                status_code=401)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)


def protect(app, *, port: int, kind: str):
    """Wrap a listening app; secret failures propagate before the server starts."""
    if kind not in {"meeting", "library"}:
        raise ValueError("Unknown UI server kind")
    return _Guard(app, port=port, kind=kind, token=load_token())


class ServerConfig(uvicorn.Config):
    """Protect the ASGI entry Uvicorn executes, before it binds its socket.

    ``app`` remains the app builder's result for in-process callers. Uvicorn
    always calls ``load`` before serving and executes only ``loaded_app``.
    """

    def __init__(self, app, *, host: str, port: int, kind: str, **kwargs):
        require_loopback(host)
        if kind not in {"meeting", "library"}:
            raise ValueError("Unknown UI server kind")
        self.ui_kind = kind
        super().__init__(app, host=host, port=port, **kwargs)

    def load(self) -> None:
        # Read first: a failed secret must not leave a loaded, unguarded config.
        token = load_token()
        super().load()
        self.loaded_app = _Guard(self.loaded_app, port=self.port, kind=self.ui_kind, token=token)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(request, timeout):
    # Never forward the secret to a redirect target or an HTTP proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


def _request(port: int, host: str, path: str, token: str | None = None):
    require_loopback(host)
    authority = f"[{host}]:{port}" if host == "::1" else f"{host}:{port}"
    headers = {"Host": f"localhost:{port}" if host == "::1" else authority}
    if token is not None:
        headers["X-Meeting-Token"] = token
    return urllib.request.Request(f"http://{authority}{path}", headers=headers)


def probe(port: int, host: str = "127.0.0.1", timeout: float = 0.6) -> str | None:
    """Verify a proof bound to the queried port and kind, without sending a secret."""
    token = load_token()
    nonce = secrets.token_hex(32)
    try:
        with _open(_request(port, host, f"{PING_PATH}?nonce={nonce}"), timeout) as response:
            payload = json.loads(response.read(4096).decode("utf-8"))
        if (isinstance(payload, dict) and payload.get("app") == "meeting-copilot"
                and payload.get("kind") in {"meeting", "library"}
                and isinstance(payload.get("proof"), str)
                and _matches(payload["proof"], _proof(token, nonce, port=port, kind=payload["kind"]))):
            return payload["kind"]
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def verified_open(port: int, path: str, *, host: str = "127.0.0.1", kind: str | None = None,
                  timeout: float = 0.6):
    """Open a local endpoint with a token only after a successful proof check."""
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Expected a local absolute path")
    found = probe(port, host=host, timeout=timeout)
    if found is None or (kind is not None and found != kind):
        raise OSError("Unverified UI server")
    return _open(_request(port, host, path, load_token()), timeout)
