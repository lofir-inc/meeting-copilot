"""Local UI authorization and discovery, without starting listening servers."""

import asyncio
import hashlib
import hmac
import io
import json
import os
import re
import time
import stat
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import running
from src.ui import guard
from src.ui.bus import EventBus
from src.ui.library import Library, build_library_router
from src.ui.server import UiConfig, build_app, start_ui_server


@pytest.fixture(autouse=True)
def isolated_token(tmp_path, monkeypatch):
    path = tmp_path / "config" / "ui-token"
    monkeypatch.setattr(guard, "TOKEN_PATH", path)
    return path


@pytest.fixture
def tiny_app():
    app = FastAPI()

    @app.get("/")
    @app.post("/api/finish")
    def action():
        return {"ok": True}

    return app


@pytest.mark.parametrize("kind", ["meeting", "library"])
def test_all_routes_require_authentication(tmp_path, kind):
    library = Library(tmp_path, lambda: {})
    router = build_library_router(library)
    if kind == "meeting":
        app = build_app(EventBus(), SimpleNamespace(), tmp_path, library_router=router)
    else:
        app = FastAPI()
        app.include_router(router)

    with TestClient(guard.protect(app, port=18965, kind=kind), base_url="http://127.0.0.1:18965") as client:
        def all_routes(router, prefix=""):
            for route in router.routes:
                if hasattr(route, "original_router"):
                    yield from all_routes(route.original_router, prefix + route.include_context.prefix)
                elif hasattr(route, "routes"):
                    yield from all_routes(route, prefix + route.path)
                else:
                    yield prefix + route.path, getattr(route, "methods", None) or {"GET"}

        routes = list(all_routes(app))
        assert len(routes) >= 40
        for path, methods in routes:
            path = re.sub(r"\{[^}]+\}", "example", path)
            for method in methods:
                response = client.request(method, path, follow_redirects=False)
                assert response.status_code == 401, (method, path, response.status_code)
        for method in ("GET", "POST", "OPTIONS", "DELETE"):
            assert client.request(method, "/unknown").status_code == 401
        assert client.get("/api/ui-ping?nonce=abc123").status_code == 200
        assert client.post("/api/ui-ping?nonce=abc123").status_code == 401


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example:18965"},
    {"Host": "127.0.0.1:3000"},
    {"Host": "testserver"},
    {"Origin": "http://127.0.0.1:3000"},
    {"Origin": "null"},
    {"Origin": "https://127.0.0.1:18965"},
    {"Sec-Fetch-Site": "cross-site"},
])
@pytest.mark.parametrize("method,path", [("POST", "/api/finish"), ("GET", "/"), ("GET", "/api/ui-ping?nonce=123")])
def test_host_and_origin_checked_before_authentication(tiny_app, headers, method, path):
    token = guard.load_token()
    with TestClient(guard.protect(tiny_app, port=18965, kind="meeting"), base_url="http://127.0.0.1:18965") as client:
        response = client.request(method, path, headers={"X-Meeting-Token": token, **headers})
        assert response.status_code == 403


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_cookie_entry_and_token_header(tiny_app, host):
    token = guard.load_token()
    base = f"http://{host}:18965"
    with TestClient(guard.protect(tiny_app, port=18965, kind="meeting"), base_url=base) as client:
        assert client.get("/").status_code == 401
        assert client.get("/", headers={"Accept": "text/html"}).status_code == 401
        assert "メニューバーか会議アシスタント.command" in client.get("/", headers={"Accept": "text/html"}).text
        assert client.get("/", headers={"X-Meeting-Token": token}).status_code == 200
        assert client.get("/", headers={b"X-Meeting-Token": "é".encode()}).status_code == 401
        entry = client.get(f"/?a=one&token={token}&a=two&empty=", follow_redirects=False)
        assert entry.status_code == 303
        assert entry.headers["location"] == base + "/?a=one&a=two&empty="
        cookie = entry.headers["set-cookie"]
        for attribute in ("HttpOnly", "SameSite=strict", "Path=/", "Max-Age=2592000"):
            assert attribute in cookie
        assert "domain=" not in cookie.lower()
        assert client.post("/api/finish", headers={"Origin": base}).status_code == 200
        # An invalid entry token must not be masked by an already valid cookie.
        assert client.get("/?token=wrong").status_code == 401
        assert client.get(f"/?token={token}&token={token}").status_code == 401


def test_browser_url_keeps_queries_and_fragments():
    url = guard.browser_url("http://127.0.0.1:18965/app/?session=meeting&token=old&tab=a#part")
    parts = urlsplit(url)
    assert parse_qs(parts.query) == {"session": ["meeting"], "token": [guard.load_token()], "tab": ["a"]}
    assert parts.fragment == "part"
    for external in ("https://evil.example/", "http://user@localhost:123/", "https://localhost:123/"):
        with pytest.raises(ValueError):
            guard.browser_url(external)


def test_ping_proof_and_invalid_nonces(tiny_app):
    token = guard.load_token()
    with TestClient(guard.protect(tiny_app, port=18965, kind="library"), base_url="http://127.0.0.1:18965") as client:
        payload = client.get("/api/ui-ping?nonce=123abc").json()
        proof = hmac.new(token.encode(), b"meeting-copilot-ui:library:18965:123abc", hashlib.sha256).hexdigest()
        assert payload == {"app": "meeting-copilot", "kind": "library", "proof": proof}
        for query in ("", "?nonce=", "?nonce=nonhex", "?nonce=12&nonce=34", "?nonce=" + "a" * 257):
            assert client.get("/api/ui-ping" + query).status_code == 400


def test_private_token_creation_repair_and_concurrent_readers(isolated_token):
    with ThreadPoolExecutor(max_workers=12) as pool:
        tokens = list(pool.map(lambda _: guard.load_token(), range(30)))
    assert len(set(tokens)) == 1
    assert stat.S_IMODE(isolated_token.stat().st_mode) == 0o600
    assert stat.S_IMODE(isolated_token.parent.stat().st_mode) == 0o700
    isolated_token.chmod(0o644)
    isolated_token.parent.chmod(0o755)
    assert guard.load_token() == tokens[0]
    assert stat.S_IMODE(isolated_token.stat().st_mode) == 0o600
    assert stat.S_IMODE(isolated_token.parent.stat().st_mode) == 0o700


def test_symlink_and_invalid_tokens_fail_closed(tmp_path, tiny_app, isolated_token):
    isolated_token.parent.mkdir()
    target = tmp_path / "other-secret"
    target.write_text("a" * 43)
    isolated_token.symlink_to(target)
    with pytest.raises(OSError):
        guard.protect(tiny_app, port=18965, kind="meeting")
    isolated_token.unlink()
    isolated_token.write_text("invalid")
    with pytest.raises(ValueError):
        guard.protect(tiny_app, port=18965, kind="meeting")


def test_unreadable_secret_prevents_server_start(tiny_app, monkeypatch):
    from src.ui import server

    def unreadable(*args, **kwargs):
        raise PermissionError("unreadable token")

    monkeypatch.setattr(guard, "load_token", unreadable)
    monkeypatch.setattr(server, "ensure_port_free", lambda *args: None)
    with pytest.raises(PermissionError, match="unreadable token"):
        start_ui_server(tiny_app, UiConfig(port=18965))
    config = guard.ServerConfig(tiny_app, host="127.0.0.1", port=18965, kind="library")
    with pytest.raises(PermissionError, match="unreadable token"):
        config.load()
    assert not config.loaded


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.1", "evil.example"])
def test_non_loopback_binding_refused(tiny_app, host):
    with pytest.raises(ValueError):
        start_ui_server(tiny_app, UiConfig(host=host, port=18965))
    with pytest.raises(ValueError):
        guard.ServerConfig(tiny_app, host=host, port=18965, kind="library")


def test_uvicorn_execution_is_guarded(tiny_app):
    config = guard.ServerConfig(tiny_app, host="127.0.0.1", port=18965, kind="library", log_level="warning")
    config.load()
    with TestClient(config.loaded_app, base_url="http://127.0.0.1:18965") as client:
        assert client.get("/").status_code == 401
        assert client.get("/", headers={"X-Meeting-Token": guard.load_token()}).status_code == 200
        assert client.get("/api/ui-ping?nonce=123").json()["kind"] == "library"
    with TestClient(config.loaded_app) as client:
        assert client.get("/").status_code == 403


def mock_peer(monkeypatch, *, kind="meeting", genuine=False):
    requests = []

    def open_request(request, timeout):
        requests.append(request)
        parts = urlsplit(request.full_url)
        if parts.path == guard.PING_PATH:
            nonce = parse_qs(parts.query)["nonce"][0]
            message = f"meeting-copilot-ui:{kind}:{parts.port}:{nonce}"
            proof = hmac.new(guard.load_token().encode(), message.encode(), hashlib.sha256).hexdigest()
            return io.BytesIO(json.dumps({"app": "meeting-copilot", "kind": kind,
                                          "proof": proof if genuine else "0" * 64}).encode())
        if parts.path == "/api/status":
            return io.BytesIO(b'{"session_name":"meeting"}')
        response = io.BytesIO(b"page")
        response.status = 200
        return response

    monkeypatch.setattr(guard, "_open", open_request)
    return requests


@pytest.mark.parametrize("real_port,real_kind,claimed_kind", [
    (18966, "library", "meeting"),  # Proofs must match both port and kind.
    (18966, "meeting", "meeting"),  # Proofs must match the port.
    (18965, "library", "meeting"),  # Proofs must match the kind.
])
def test_relayed_guard_proof_is_never_trusted_or_sent_secret(tiny_app, monkeypatch,
                                                           real_port, real_kind, claimed_kind):
    fake_port = 18965
    requests = []
    # Exercise the real ASGI Guard at its own port without binding listening sockets.
    with TestClient(guard.protect(tiny_app, port=real_port, kind=real_kind),
                    base_url=f"http://127.0.0.1:{real_port}") as genuine:
        def relay(request, timeout):
            requests.append(request)
            parts = urlsplit(request.full_url)
            assert parts.port == fake_port
            if parts.path == guard.PING_PATH:
                response = genuine.get(f"{parts.path}?{parts.query}")
                assert response.status_code == 200
                payload = response.json()
                assert payload["kind"] == real_kind
                payload["kind"] = claimed_kind
                return io.BytesIO(json.dumps(payload).encode())
            return io.BytesIO(b'{"session_name":"forged meeting"}')

        monkeypatch.setattr(guard, "_open", relay)
        assert guard.probe(fake_port) is None
        assert running.find_meeting(ports=[fake_port], is_open=lambda port: True) is None
        for expected_kind in (None, "meeting"):
            with pytest.raises(OSError, match="Unverified UI server"):
                guard.verified_open(fake_port, "/api/status", kind=expected_kind)

    assert len(requests) == 4
    token = guard.load_token()
    for request in requests:
        assert urlsplit(request.full_url).path == guard.PING_PATH
        assert not request.has_header("X-meeting-token")
        assert token not in request.full_url
        assert token not in str(request.header_items())
        assert request.data is None


def test_fake_peer_is_never_trusted_or_sent_secret(tmp_path, monkeypatch):
    requests = mock_peer(monkeypatch)
    assert guard.probe(18965) is None
    assert running.find_meeting(ports=[18965], is_open=lambda port: True) is None
    library = Library(tmp_path, lambda: {})
    library.MEETING_PORT, library.MEETING_PORTS = 18965, 1
    assert library.meeting_status()["running"] is False
    assert requests
    for request in requests:
        assert urlsplit(request.full_url).path == guard.PING_PATH
        assert not request.has_header("X-meeting-token")
        assert guard.load_token() not in request.full_url
        assert guard.load_token() not in str(request.header_items())


def test_verified_meeting_is_queried_with_token(tmp_path, monkeypatch):
    requests = mock_peer(monkeypatch, genuine=True)
    assert guard.probe(18965) == "meeting"
    assert running.find_meeting(ports=[18965], is_open=lambda port: True).session_name == "meeting"
    library = Library(tmp_path, lambda: {})
    library.MEETING_PORT, library.MEETING_PORTS = 18965, 1
    assert library.meeting_status()["running"]
    for index, request in enumerate(requests):
        if urlsplit(request.full_url).path == guard.PING_PATH:
            assert not request.has_header("X-meeting-token")
        else:
            assert request.get_header("X-meeting-token") == guard.load_token()
            assert urlsplit(requests[index - 1].full_url).path == guard.PING_PATH




@pytest.mark.parametrize("already_running,has_page", [(False, True), (True, True), (True, False)])
def test_no_browser_prints_authenticated_entry(monkeypatch, capsys, already_running, has_page):
    from scripts import library_ui

    port = 18966
    monkeypatch.setattr(library_ui.sys, "argv", ["library_ui.py", "--no-browser", "--port", str(port),
                                              "--session", "example session", "--tab", "dictionary"])
    monkeypatch.setattr(library_ui, "ensure_user_path", lambda: None)
    monkeypatch.setattr(library_ui, "port_in_use", lambda *args: already_running)
    monkeypatch.setattr(guard, "probe", lambda *args, **kwargs: "library")
    monkeypatch.setattr(library_ui, "task_hub_pusher", lambda *args: None)
    monkeypatch.setattr(library_ui, "threading", SimpleNamespace(
        Thread=lambda **kwargs: SimpleNamespace(start=lambda: None),
    ))
    monkeypatch.setattr(library_ui.uvicorn, "Server", lambda config: SimpleNamespace(run=lambda: None))
    monkeypatch.setattr(library_ui.webbrowser, "open", lambda *args: pytest.fail("--no-browser must not open a browser"))

    assert library_ui.main() == 0
    printed = capsys.readouterr().out
    url = re.search(r"http://[^\s（]+", printed).group()
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.netloc == f"127.0.0.1:{port}"
    assert parts.path == "/library"
    assert query["session"] == ["example session"]
    assert query["token"] == [guard.load_token()]
    # The printed URL must actually establish a session when opened by hand.
    with TestClient(guard.protect(FastAPI(), port=port, kind="library"),
                    base_url=f"http://127.0.0.1:{port}") as client:
        entry = client.get(url, follow_redirects=False)
        assert entry.status_code == 303
        assert client.cookies["mc_ui"] == guard.load_token()


def test_authenticated_requests_do_not_follow_redirects(monkeypatch):
    handler = guard._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://evil.example/") is None
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            raise urllib.error.HTTPError(request.full_url, 302, "Found", {}, None)

    def build_opener(*handlers):
        assert isinstance(handlers[0], guard.urllib.request.ProxyHandler)
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], guard._NoRedirect)
        return Opener()

    monkeypatch.setattr(guard.urllib.request, "build_opener", build_opener)
    assert guard.probe(18965) is None
    assert len(calls) == 1
    assert not calls[0][0].has_header("X-meeting-token")


def test_events_are_forwarded_before_the_stream_finishes(tmp_path):
    async def exercise():
        bus = EventBus()
        app = guard.protect(build_app(bus, SimpleNamespace(), tmp_path), port=18965, kind="meeting")
        first = asyncio.Event()
        messages = []
        bus.publish("segment", {"text": "first"})

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and b"first" in message.get("body", b""):
                first.set()

        async def receive():
            await asyncio.Event().wait()

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/events",
                 "raw_path": b"/events", "query_string": b"", "root_path": "",
                 "server": ("127.0.0.1", 18965), "client": ("127.0.0.1", 1234),
                 "headers": [(b"host", b"127.0.0.1:18965"),
                             (b"cookie", f"mc_ui={guard.load_token()}".encode())]}
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(first.wait(), timeout=1)
            assert not task.done()
            assert messages[0]["type"] == "http.response.start"
            assert messages[0]["status"] == 200
            assert not any(item["type"] == "http.response.body" and not item.get("more_body") for item in messages)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(exercise())


def test_作る途中で空のまま残ったトークンは作り直す(isolated_token):
    isolated_token.parent.mkdir(mode=0o700, parents=True)
    isolated_token.touch(mode=0o600)
    old = time.time() - 60
    os.utime(isolated_token, (old, old))
    token = guard.load_token()
    assert guard._TOKEN.fullmatch(token)
    assert isolated_token.read_text().strip() == token


def test_作っている最中の空のトークンは消さない(isolated_token, monkeypatch):
    isolated_token.parent.mkdir(mode=0o700, parents=True)
    isolated_token.touch(mode=0o600)
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    with pytest.raises(ValueError):
        guard.load_token()
    assert isolated_token.exists()


def test_unverified_library_uses_spare_port(tmp_path, monkeypatch, capsys):
    from scripts import library_ui

    monkeypatch.setattr(library_ui.sys, "argv", ["library_ui.py", "--no-browser", "--port", "18966"])
    monkeypatch.setattr(library_ui, "REPO", tmp_path)
    monkeypatch.setattr(library_ui, "ensure_user_path", lambda: None)
    monkeypatch.setattr(library_ui, "port_in_use", lambda host, port: port == 18966)
    monkeypatch.setattr(guard, "probe", lambda *args, **kwargs: None)
    monkeypatch.setattr(library_ui, "task_hub_pusher", lambda *args: None)
    monkeypatch.setattr(library_ui, "threading", SimpleNamespace(
        Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)))
    configs = []
    monkeypatch.setattr(library_ui.uvicorn, "Server", lambda config: (
        configs.append(config) or SimpleNamespace(run=lambda: None)))
    assert library_ui.main() == 0
    assert configs[0].port == 8776
    assert "127.0.0.1:8776/" in capsys.readouterr().out


def test_menubar_does_not_open_unverified_library(tmp_path, monkeypatch):
    from src.app import menubar

    monkeypatch.setattr(running, "port_open", lambda port: True)
    monkeypatch.setattr(guard, "probe", lambda port: None)
    monkeypatch.setattr(menubar.webbrowser, "open", lambda url: pytest.fail("Unverified browser target"))
    launched = []
    monkeypatch.setattr(menubar.subprocess, "Popen", lambda command: launched.append(command))
    menubar.open_library(tmp_path, tmp_path / "python")
    assert launched[0][1] == str(tmp_path / "scripts" / "library_ui.py")
