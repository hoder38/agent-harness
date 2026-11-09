from app.channels import line, notify as notify_mod, telegram
from app.settings import get_settings


def _settings(monkeypatch, *, tg: bool, ln: bool = True):
    s = get_settings()
    monkeypatch.setattr(s, "telegram_bot_token", "t" if tg else "")
    monkeypatch.setattr(s, "telegram_owner_chat_id", "42" if tg else "")
    monkeypatch.setattr(s, "line_owner_user_id", "Uowner" if ln else "")
    monkeypatch.setattr(s, "line_channel_access_token", "tok" if ln else "")


def _record(monkeypatch):
    sent = []

    async def tg(text):
        sent.append(("telegram", text))

    async def push(to, text):
        sent.append(("push", text))

    monkeypatch.setattr(telegram, "send", tg)
    monkeypatch.setattr(line, "push", push)
    return sent


async def test_notify_prefers_telegram(monkeypatch):
    _settings(monkeypatch, tg=True)
    sent = _record(monkeypatch)
    assert await notify_mod.notify("hi")
    assert sent == [("telegram", "hi")]


async def test_notify_falls_back_to_line_push_then_nothing(monkeypatch):
    _settings(monkeypatch, tg=False)
    sent = _record(monkeypatch)
    assert await notify_mod.notify("hi")
    _settings(monkeypatch, tg=False, ln=False)
    assert not await notify_mod.notify("again")
    assert sent == [("push", "hi")]


async def test_notify_never_raises(monkeypatch):
    _settings(monkeypatch, tg=True)

    async def boom(text):
        raise RuntimeError("down")

    monkeypatch.setattr(telegram, "send", boom)
    assert not await notify_mod.notify("hi")


async def test_deliver_uses_telegram_instead_of_line_push(monkeypatch):
    _settings(monkeypatch, tg=True)
    sent = _record(monkeypatch)
    assert await line.deliver("Uowner", "answer", None, None, 50) == "telegram"
    assert sent == [("telegram", "answer")]


async def test_deliver_never_pushes_a_missed_window(monkeypatch):
    _settings(monkeypatch, tg=False)
    sent = _record(monkeypatch)
    assert await line.deliver("Uowner", "answer", "R", 1.0, 50) == "dropped"
    assert sent == []
