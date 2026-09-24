"""外部送信の関門の単体テスト。

この関門が緩むと、2026-09-13 の事故（クライアント会議 50 分を無料枠へ送り、消せなくなった）が
そのまま再発する。**通らないケースを潰すテスト**が本体で、通るケースは 1 本でよい。
"""

from pathlib import Path

import pytest

from src.stt.external_consent import (
    Approval,
    ExternalSendRefused,
    ExternalSttConfig,
    SEND_LOG,
    approve,
    LOGIN_EXPIRED,
    billing_enabled,
    record_send,
)


def _config(**overrides) -> ExternalSttConfig:
    values = {"enabled": True, "billing_project": "your-gcp-project"}
    values.update(overrides)
    return ExternalSttConfig(**values)


def _paid(monkeypatch, enabled: bool = True, reason: str = "課金が有効です") -> None:
    monkeypatch.setattr("src.stt.external_consent.billing_enabled",
                        lambda project: (enabled, reason))


class TestApprove:
    def test_承認と課金の両方が揃えば通る(self, tmp_path, monkeypatch):
        _paid(monkeypatch)
        approval = approve(_config(), tmp_path, [], ask=lambda prompt: True)
        assert approval.billing_project == "your-gcp-project"
        assert approval.at

    def test_設定が_off_なら聞きもしない(self, tmp_path, monkeypatch):
        _paid(monkeypatch)
        asked = []
        with pytest.raises(ExternalSendRefused):
            approve(_config(enabled=False), tmp_path, [], ask=lambda prompt: asked.append(prompt) or True)
        assert asked == []

    def test_プロジェクトが空なら聞きもしない(self, tmp_path, monkeypatch):
        """送り先を固定していないのに送らない（運用者 決定 2026-09-13: 専用プロジェクトを 1 つ）。"""
        _paid(monkeypatch)
        asked = []
        with pytest.raises(ExternalSendRefused):
            approve(_config(billing_project=""), tmp_path, [],
                    ask=lambda prompt: asked.append(prompt) or True)
        assert asked == []

    def test_承認しなければ課金を確かめるまでもなく止まる(self, tmp_path, monkeypatch):
        checked = []
        monkeypatch.setattr("src.stt.external_consent.billing_enabled",
                            lambda project: checked.append(project) or (True, ""))
        with pytest.raises(ExternalSendRefused):
            approve(_config(), tmp_path, [], ask=lambda prompt: False)
        assert checked == []

    def test_課金が確認できなければ承認済みでも止まる(self, tmp_path, monkeypatch):
        """オフライン・gcloud の認証切れもここに来る（fail-closed）。"""
        _paid(monkeypatch, enabled=False, reason="課金状態を確認できません")
        with pytest.raises(ExternalSendRefused, match="確認できない"):
            approve(_config(), tmp_path, [], ask=lambda prompt: True)

    def test_画面が無ければ送らない(self, tmp_path, monkeypatch):
        """TTY でなければ会議ごとの確認が取れない ＝ 送らない。

        理由が「断られた」ではなく「聞けなかった」と分かる文言であること。
        同じ文だと「n と答えたつもりが、実は何も聞かれていなかった」に気づけない
        （2026-09-13 に実際に紛らわしかった）。
        """
        _paid(monkeypatch)

        class NotATty:
            def isatty(self):
                return False

        with pytest.raises(ExternalSendRefused, match="聞けません"):
            approve(_config(), tmp_path, [], stream=NotATty())

    def test_確認の文面に会議名と送り先が出る(self, tmp_path, monkeypatch):
        _paid(monkeypatch)
        seen = []
        approve(_config(), tmp_path, [Path("recording_remote.wav")],
                ask=lambda prompt: seen.append(prompt) or True)
        assert tmp_path.name in seen[0]
        assert "your-gcp-project" in seen[0]
        assert "recording_remote.wav" in seen[0]


class TestRecordSend:
    def test_送った記録が残る(self, tmp_path):
        """09-13 の事故は「送ったこと自体が後から辿れない」のがいちばん痛かった。"""
        approval = Approval(billing_project="your-gcp-project",
                            files=[Path("recording_remote.wav")], at="2026-09-13T21:00:00+09:00")
        path = record_send(tmp_path, approval, model="gemini-3.5-transcribe", note="finalize")
        assert path.name == SEND_LOG
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "your-gcp-project" in lines[0]
        assert "recording_remote.wav" in lines[0]

    def test_二回送れば二行になる(self, tmp_path):
        approval = Approval(billing_project="p", files=[], at="2026-09-13T21:00:00+09:00")
        record_send(tmp_path, approval, model="m")
        path = record_send(tmp_path, approval, model="m")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


class TestConfig:
    def test_知らないキーは捨てる(self):
        config = ExternalSttConfig.from_mapping({"enabled": True, "未知のキー": 1})
        assert config.enabled is True

    def test_既定は_off(self):
        assert ExternalSttConfig.from_mapping(None).enabled is False
        assert ExternalSttConfig().billing_project == ""


class TestAskTimeout:
    """会議後の作り直しは無人で走る。返事を待ち続けると議事録ができない。"""

    def _tty(self, monkeypatch, ready: bool):
        import src.stt.external_consent as module

        class FakeTty:
            def isatty(self):
                return True

            def readline(self):
                return "y\n"

        monkeypatch.setattr(module.select, "select",
                            lambda r, w, x, t: (([FakeTty()], [], []) if ready else ([], [], [])))
        return FakeTty()

    def test_返事が無ければ送らない(self, tmp_path, monkeypatch, capsys):
        _paid(monkeypatch)
        stream = self._tty(monkeypatch, ready=False)
        with pytest.raises(ExternalSendRefused):
            approve(_config(ask_timeout_sec=1.0), tmp_path, [], stream=stream)
        assert "返事が無いので送りません" in capsys.readouterr().out

    def test_時間内に_y_なら通る(self, tmp_path, monkeypatch):
        _paid(monkeypatch)
        stream = self._tty(monkeypatch, ready=True)
        assert approve(_config(ask_timeout_sec=1.0), tmp_path, [], stream=stream).billing_project


# ------------------------------------------ 会議前の点検で見る「確認できない」の形

def test_認証切れは直し方の分かる文にする(monkeypatch):
    """実際に起きた（2026-09-14）。放っておくと会議中に黙って手元へ落ちる。"""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 1, stdout="", stderr="ERROR: (gcloud.billing.projects.describe) There was a problem "
                                   "refreshing your current auth tokens: Reauthentication failed."))
    ok, reason = billing_enabled("your-gcp-project")

    assert ok is False
    assert "gcloud auth login" in reason


def test_長い_gcloud_エラーは1行に畳む(monkeypatch):
    """画面に出すので、改行だらけ・途中で切れた文のまま出さない。"""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 1, stdout="", stderr="ERROR: (gcloud.billing.projects.describe) 権限がありません\n" + "あ" * 300))
    ok, reason = billing_enabled("your-gcp-project")

    assert ok is False
    assert "\n" not in reason
    assert "gcloud.billing" not in reason        # コマンド名の前置きは落とす
    assert reason.endswith("…")


def test_短い待ち時間を渡せる(monkeypatch):
    """会議前の点検は起動を長く止めない。"""
    import subprocess

    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: seen.update(kwargs) or
                        subprocess.CompletedProcess(args, 0, stdout="True", stderr=""))
    assert billing_enabled("your-gcp-project", timeout_sec=20.0)[0] is True
    assert seen["timeout"] == 20.0


class _Process:
    def __init__(self, hang=False):
        self.hang = hang
        self.killed = False

    def wait(self, timeout=None):
        import subprocess
        if self.hang:
            raise subprocess.TimeoutExpired("gcloud", timeout)
        return 0

    def kill(self):
        self.killed = True


def test_ログイン画面を開いて終わったら確かめ直す(monkeypatch):
    import subprocess
    from src.stt.external_consent import login_and_wait

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 0, stdout="you@example.com\n", stderr=""))
    launched, said = [], []

    ok, _ = login_and_wait("your-gcp-project", on_status=said.append,
                           popen=lambda command, **kwargs: launched.append(command) or _Process(),
                           check=lambda: (True, "課金が有効です"))

    assert ok is True
    assert launched == [["gcloud", "auth", "login", "--brief", "you@example.com"]]   # 前回のアカウントを指定
    assert "ブラウザで開きました" in said[0] and "ログインできました" in said[-1]


def test_ログインが終わらなければ送らない側に倒す(monkeypatch):
    import subprocess
    from src.stt.external_consent import login_and_wait

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    process = _Process(hang=True)

    ok, reason = login_and_wait("p", timeout_sec=1, popen=lambda command, **kwargs: process,
                                check=lambda: (True, "呼ばれないはず"))

    assert ok is False and process.killed and "終わりませんでした" in reason


def test_アカウントが無いときも認証切れとして扱う(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args, 1, stdout="", stderr="ERROR: (gcloud.billing.projects.describe) You do not currently have an active account selected."))

    assert billing_enabled("p")[1] == LOGIN_EXPIRED


def test_確認専用のアカウントの鍵があれば先にそれで聞く(monkeypatch, tmp_path):
    """人のログインは 16 時間ほどで切れる。サービスアカウントの鍵は切れない。"""
    import subprocess

    key = tmp_path / "billing.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEETING_BILLING_CREDENTIAL", str(key))
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: seen.append((kwargs.get("env") or {}).get(
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE")) or subprocess.CompletedProcess(args, 0, stdout="True", stderr=""))

    ok, reason = billing_enabled("p")

    assert ok is True and "確認専用" in reason and seen == [str(key)]


def test_鍵で確かめられなければ人のログインで聞き直す(monkeypatch, tmp_path):
    import subprocess

    key = tmp_path / "billing.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEETING_BILLING_CREDENTIAL", str(key))
    calls = []

    def run(args, **kwargs):
        with_key = bool((kwargs.get("env") or {}).get("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"))
        calls.append(with_key)
        if with_key:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="ERROR: 権限がありません")
        return subprocess.CompletedProcess(args, 0, stdout="True", stderr="")

    monkeypatch.setattr(subprocess, "run", run)

    assert billing_enabled("p") == (True, "課金が有効です") and calls == [True, False]


def test_鍵が無ければ今までどおり(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setenv("MEETING_BILLING_CREDENTIAL", str(tmp_path / "無い.json"))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(
        args, 1, stdout="", stderr="Reauthentication failed."))

    assert billing_enabled("p")[1] == LOGIN_EXPIRED
