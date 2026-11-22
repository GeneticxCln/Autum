"""Run Autum in the background: a bubble in the screen corner is her only face.

    PYTHONPATH=src python3 -m autum

Hold the talk key (KDE asks you to confirm it the first time) or hold the
bubble, speak, and let go. Pressing it while Autum speaks interrupts her.
Everything runs locally: Ollama on localhost and PipeWire for audio.
Nothing is recorded to disk. Stop Autum with Ctrl+C or by ending the process.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import os
import shutil
import signal
import sys

import httpx

from autum.audio import MIC_HELP, MuteMonitor, SpeechPlayer, VoiceInput, check_microphone
from autum.bus import EventBus
from autum.events import new_id
from autum.hotkey import PushToTalkKey
from autum.ollama import OmniChat, OmniSpeech, OmniTranscriber, check_ready, preload
from autum.runtime import Mode, Runtime
from autum.transport import EventServer

PERSONA = (
    "You are Autum, a friendly voice assistant running locally on the user's computer. "
    "Your replies are spoken aloud, so answer in one to three short sentences of plain "
    "conversational text, with no lists, headings, or markdown. You cannot see the "
    "screen or take actions yet; if asked to, say so briefly."
)


def log(message: str) -> None:
    print(f"autum: {message}", file=sys.stderr, flush=True)


def system_prompt() -> str:
    """Built per turn, so Autum knows the real time instead of guessing it."""
    now = dt.datetime.now().astimezone()
    return (
        f"{PERSONA} The current local date and time is {now:%A %d %B %Y, %H:%M} ({now.tzname()}). "
        "Mention it only when asked, and never guess a different time or date."
    )


def command_handler(runtime: Runtime, voice: VoiceInput):
    """User actions from the bubble (validated by the transport)."""

    async def handle(command: dict) -> None:
        match command["command"]:
            case "talk_start":
                await voice.start()
            case "talk_finish":
                await voice.finish()
            case "stop":
                await voice.abort()
                await runtime.cancel()
            case "text":
                await voice.abort()
                runtime.submit_text(command["text"])

    return handle


async def start_bubble() -> asyncio.subprocess.Process | None:
    """The bubble runs as its own process; Autum keeps working without it."""
    if importlib.util.find_spec("PySide6") is None:
        log("the bubble needs PySide6, which is not installed")
        return None
    env = dict(os.environ)
    from autum.presence import has_layer_shell

    if env.get("WAYLAND_DISPLAY") and has_layer_shell():
        # Native Wayland with wlr-layer-shell pins the cloud to the corner
        # across all virtual desktops and windows without taskbar entries.
        env["QT_QPA_PLATFORM"] = "wayland"
    elif env.get("DISPLAY"):
        # Fallback for X11 or Wayland compositors without layer-shell-qt.
        env["QT_QPA_PLATFORM"] = "xcb"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "autum.presence",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, env=env,
    )
    try:
        await asyncio.wait_for(proc.wait(), 1.5)
        log("the bubble could not start")
        return None
    except asyncio.TimeoutError:
        return proc  # still running: it started


async def main() -> int:
    missing = [tool for tool in ("pw-record", "pw-play", "wpctl") if not shutil.which(tool)]
    if missing:
        log(f"missing PipeWire tools: {', '.join(missing)}")
        return 1
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=3.0)) as client:
        problem = await check_ready(client)
        if problem:
            log(problem)
            return 1
        if not await check_microphone():
            log(MIC_HELP)
            return 1
        await preload(client)

        bus = EventBus(new_id())
        bus.subscribe(
            lambda e: log(f"{e.payload['component']}: {e.payload['error']}"),
            {"runtime.component_error"},
        )
        speech = SpeechPlayer(bus, OmniSpeech(client))
        runtime = Runtime(bus, OmniChat(client), speech, mode=Mode.PUSH_TO_TALK, system_prompt=system_prompt)
        voice = VoiceInput(bus, runtime, OmniTranscriber(client))
        server = EventServer(bus, command_handler(runtime, voice))
        try:
            await server.start()
        except RuntimeError as exc:
            log(str(exc))
            return 1
        mute = asyncio.create_task(MuteMonitor(bus).run())
        bubble = await start_bubble()
        PushToTalkKey(
            loop,
            on_press=lambda: loop.create_task(voice.start()),
            on_release=lambda: loop.create_task(voice.finish()),
            on_status=log,
        ).start()
        log("running (provisional pipeline D). Stop with Ctrl+C.")
        try:
            await stop.wait()
        finally:
            if bubble is not None and bubble.returncode is None:
                bubble.terminate()
                await bubble.wait()
            await server.close()
            mute.cancel()
            await voice.abort()
            await runtime.cancel()
            speech.stop()
            try:
                await asyncio.wait_for(runtime.wait_until_settled(), 3)
            except asyncio.TimeoutError:
                pass
    return 0
