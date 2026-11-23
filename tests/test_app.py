"""Commands from the presence window map to the right actions."""

import asyncio

from autum.app import command_handler
from autum.runtime import Mode


class FakeVoice:
    def __init__(self):
        self.calls = []

    async def start(self):
        self.calls.append("start")

    async def finish(self):
        self.calls.append("finish")

    async def abort(self):
        self.calls.append("abort")


class FakeHandsFree:
    def __init__(self):
        self.calls = []

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.mode = Mode.PUSH_TO_TALK

    async def cancel(self, source="user_command"):
        self.calls.append(("cancel", source))

    def submit_text(self, text):
        self.calls.append(("text", text))

    def set_mode(self, mode):
        self.mode = mode
        self.calls.append(("mode", mode))

    def resolve_approval(self, response, **kwargs):
        self.calls.append(("approval", response, kwargs.get("source")))


def test_presence_commands():
    voice, runtime = FakeVoice(), FakeRuntime()
    handle = command_handler(runtime, voice)

    async def scenario():
        for command in [{"command": "talk_start"}, {"command": "talk_finish"}, {"command": "stop"},
                        {"command": "text", "text": "hi"}]:
            await handle(command)

    asyncio.run(scenario())
    assert voice.calls == ["start", "finish", "abort", "abort"]
    assert runtime.calls == [("cancel", "user_command"), ("text", "hi")]


def test_mode_switching_and_hands_free_manual_interrupt():
    voice, hf, runtime = FakeVoice(), FakeHandsFree(), FakeRuntime()
    handle = command_handler(runtime, voice, hf)

    async def scenario():
        await handle({"command": "set_mode", "mode": "hands_free"})
        await handle({"command": "talk_start"})   # interrupts in hands-free mode
        await handle({"command": "talk_finish"})  # no-op in hands-free mode
        await handle({"command": "set_mode", "mode": "push_to_talk"})
        await handle({"command": "talk_start"})
        await handle({"command": "approve"})
        await handle({"command": "deny"})

    asyncio.run(scenario())
    assert hf.calls == ["start", "stop"]
    assert voice.calls == ["abort", "start"]
    assert runtime.calls == [
        ("mode", Mode.HANDS_FREE),
        ("cancel", "push_to_talk"),
        ("mode", Mode.PUSH_TO_TALK),
        ("approval", "yes", "command"),
        ("approval", "no", "command"),
    ]


def test_reload_bubble_and_close_bubble_commands():
    voice, runtime = FakeVoice(), FakeRuntime()
    events = []

    async def on_reload():
        events.append("reload_config")

    async def on_reload_bubble():
        events.append("reload_bubble")

    def on_close_bubble():
        events.append("close_bubble")

    handle = command_handler(
        runtime,
        voice,
        on_reload=on_reload,
        on_reload_bubble=on_reload_bubble,
        on_close_bubble=on_close_bubble,
    )

    async def scenario():
        await handle({"command": "reload_bubble"})
        await handle({"command": "close_bubble"})

    asyncio.run(scenario())
    assert events == ["reload_config", "reload_bubble", "close_bubble"]
    assert voice.calls == ["abort"]
    assert runtime.calls == [("cancel", "user_command")]
