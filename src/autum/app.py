"""Run Autum in the background: a bubble in the screen corner is her only face.

    PYTHONPATH=src python3 -m autum

Hold the talk key (KDE asks you to confirm it the first time) or hold the
bubble, speak, and let go. Pressing it while Autum speaks interrupts her.
Everything runs locally: Ollama on localhost and PipeWire for audio.
Nothing is recorded to disk. Stop Autum with Ctrl+C or by ending the process.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import os
import shutil
import signal
import sys
from typing import Callable, Sequence

import httpx

from autum.audio import (
    MIC_HELP,
    EchoCanceller,
    HandsFreeInput,
    MuteMonitor,
    SpeechPlayer,
    VoiceInput,
    check_microphone,
)
from autum.bus import EventBus
from autum.events import new_id
from autum.hotkey import PushToTalkKey
from autum.ollama import OmniChat, OmniSpeech, OmniTranscriber, check_ready, preload
from autum.runtime import Mode, Runtime
from autum.transport import EventServer
from autum.vad import SileroVad, TurnDetector

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


def command_handler(
    runtime: Runtime,
    voice: VoiceInput,
    hands_free: HandsFreeInput | Callable[[], HandsFreeInput] | None = None,
):
    """User actions from the bubble (validated by the transport)."""
    hf_instance: HandsFreeInput | None = hands_free if not callable(hands_free) else None

    def get_hands_free() -> HandsFreeInput | None:
        nonlocal hf_instance
        if hf_instance is None and callable(hands_free):
            hf_instance = hands_free()
        return hf_instance

    async def handle(command: dict) -> None:
        match command["command"]:
            case "talk_start":
                if getattr(runtime, "mode", Mode.PUSH_TO_TALK) is Mode.HANDS_FREE:
                    await runtime.cancel("push_to_talk")
                else:
                    await voice.start()
            case "talk_finish":
                if getattr(runtime, "mode", Mode.PUSH_TO_TALK) is Mode.PUSH_TO_TALK:
                    await voice.finish()
            case "stop":
                await voice.abort()
                await runtime.cancel()
            case "text":
                await voice.abort()
                runtime.submit_text(command["text"])
            case "set_mode":
                target_mode = Mode(command["mode"])
                hf = get_hands_free()
                if target_mode is Mode.HANDS_FREE and hf is not None:
                    await voice.abort()
                    runtime.set_mode(Mode.HANDS_FREE)
                    await hf.start()
                elif target_mode is Mode.PUSH_TO_TALK:
                    if hf is not None:
                        await hf.stop()
                    runtime.set_mode(Mode.PUSH_TO_TALK)

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python3 -m autum")
    parser.add_argument(
        "--hands-free",
        action="store_true",
        help="Start in hands-free mode with Silero VAD turn detection and voice barge-in",
    )
    parser.add_argument(
        "--aec",
        action="store_true",
        help="Load PipeWire WebRTC echo cancellation for speaker use in hands-free mode",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=700,
        help="Hands-free end-of-turn silence threshold in milliseconds (default: 700)",
    )
    parser.add_argument(
        "--barge-in-ms",
        type=int,
        default=256,
        help="Hands-free speech duration in milliseconds required to interrupt Autum (default: 256)",
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [tool for tool in ("pw-record", "pw-play", "wpctl") if not shutil.which(tool)]
    if args.aec and not shutil.which("pactl"):
        missing.append("pactl")
    if missing:
        log(f"missing PipeWire tools: {', '.join(missing)}")
        return 1
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    aec: EchoCanceller | None = None
    in_target: str | None = None
    out_target: str | None = None
    if args.aec:
        aec = EchoCanceller()
        if await aec.start():
            in_target, out_target = aec.source, aec.sink
            log("PipeWire WebRTC echo cancellation active.")
        else:
            log("could not start PipeWire echo cancellation; continuing without AEC")
            aec = None

    try:
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
            mode = Mode.HANDS_FREE if args.hands_free else Mode.PUSH_TO_TALK
            speech = SpeechPlayer(bus, OmniSpeech(client), target=out_target)
            runtime = Runtime(bus, OmniChat(client), speech, mode=mode, system_prompt=system_prompt)
            transcriber = OmniTranscriber(client)
            voice = VoiceInput(bus, runtime, transcriber, target=in_target)

            hands_free: HandsFreeInput | None = None

            def make_hands_free() -> HandsFreeInput:
                nonlocal hands_free
                if hands_free is None:
                    detector = TurnDetector(
                        SileroVad(),
                        silence_ms=args.silence_ms,
                        barge_in_ms=args.barge_in_ms,
                    )
                    hands_free = HandsFreeInput(bus, runtime, transcriber, detector, target=in_target)
                return hands_free

            handle_command = command_handler(runtime, voice, make_hands_free)
            server = EventServer(bus, handle_command)
            try:
                await server.start()
            except RuntimeError as exc:
                log(str(exc))
                return 1
            mute = asyncio.create_task(MuteMonitor(bus).run())
            bubble = await start_bubble()
            if args.hands_free:
                await make_hands_free().start()
                if aec is None:
                    log("hands-free mode active (use headphones or pass --aec when using speakers).")
            PushToTalkKey(
                loop,
                on_press=lambda: loop.create_task(handle_command({"command": "talk_start"})),
                on_release=lambda: loop.create_task(handle_command({"command": "talk_finish"})),
                on_status=log,
            ).start()
            log(f"running ({mode.value}, provisional pipeline D). Stop with Ctrl+C.")
            try:
                await stop.wait()
            finally:
                if bubble is not None and bubble.returncode is None:
                    bubble.terminate()
                    await bubble.wait()
                await server.close()
                mute.cancel()
                if hands_free is not None:
                    await hands_free.stop()
                await voice.abort()
                await runtime.cancel()
                speech.stop()
                try:
                    await asyncio.wait_for(runtime.wait_until_settled(), 3)
                except asyncio.TimeoutError:
                    pass
    finally:
        if aec is not None:
            await aec.stop()
    return 0
