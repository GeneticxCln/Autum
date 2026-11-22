"""Commands from the presence window map to the right actions."""

import asyncio

from autum.app import command_handler


class FakeVoice:
    def __init__(self):
        self.calls = []

    async def start(self):
        self.calls.append("start")

    async def finish(self):
        self.calls.append("finish")

    async def abort(self):
        self.calls.append("abort")


class FakeRuntime:
    def __init__(self):
        self.calls = []

    async def cancel(self):
        self.calls.append("cancel")

    def submit_text(self, text):
        self.calls.append(("text", text))


def test_presence_commands():
    voice, runtime = FakeVoice(), FakeRuntime()
    handle = command_handler(runtime, voice)

    async def scenario():
        for command in [{"command": "talk_start"}, {"command": "talk_finish"}, {"command": "stop"},
                        {"command": "text", "text": "hi"}]:
            await handle(command)

    asyncio.run(scenario())
    assert voice.calls == ["start", "finish", "abort", "abort"]
    assert runtime.calls == ["cancel", ("text", "hi")]
