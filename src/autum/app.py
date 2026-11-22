"""Talk to Autum in a terminal (M2, provisional pipeline D).

    PYTHONPATH=src python3 -m autum

Space starts talking and Space again ends your turn; pressing Space while
Autum speaks interrupts it. Everything runs locally: Ollama on localhost
and PipeWire for audio. Nothing is recorded to disk.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import shutil
import sys
import termios
import tty

import httpx

from autum.audio import MIC_HELP, MuteMonitor, SpeechPlayer, VoiceInput, check_microphone
from autum.bus import EventBus
from autum.events import Event, new_id
from autum.ollama import OmniChat, OmniSpeech, OmniTranscriber, check_ready, preload
from autum.runtime import Mode, Runtime

PERSONA = (
    "You are Autum, a friendly voice assistant running locally on the user's computer. "
    "Your replies are spoken aloud, so answer in one to three short sentences of plain "
    "conversational text, with no lists, headings, or markdown. You cannot see the "
    "screen or take actions yet; if asked to, say so briefly."
)

KEYS = "Space: talk, then Space when done (also interrupts Autum)   s: stop Autum   t: type   q: quit"


def system_prompt() -> str:
    """Built per turn, so Autum knows the real time instead of guessing it."""
    now = dt.datetime.now().astimezone()
    return (
        f"{PERSONA} The current local date and time is {now:%A %d %B %Y, %H:%M} ({now.tzname()}). "
        "Mention it only when asked, and never guess a different time or date."
    )


class Screen:
    """Shows the conversation and Autum's status from runtime events."""

    def __init__(self, bus: EventBus) -> None:
        self.replying = False
        bus.subscribe(self.on_event)

    def line(self, text: str) -> None:
        if self.replying:
            print()
            self.replying = False
        print(text, flush=True)

    def on_event(self, e: Event) -> None:
        match e.type:
            case "audio.speech_started":
                if self.replying:
                    print(" [interrupted]")
                    self.replying = False
                self.line("● Listening… press Space when you're done.")
            case "conversation.response_started":
                self.line("")
                print("Autum: ", end="", flush=True)
                self.replying = True
            case "conversation.response_chunk":
                print(e.payload["text"], end="", flush=True)
            case "conversation.response_finished":
                self.line("")
            case "operation.cancelled" if self.replying:
                print(" [stopped]", flush=True)
                self.replying = False
            case "audio.transcribed" if e.payload["final"] and e.payload["text"]:
                self.line(f"  (you said: {e.payload['text']})")
            case "runtime.component_error":
                self.line(f"! {e.payload['component']}: {e.payload['error']}")
            case "audio.mute_changed":
                self.line("Microphone muted: unmute it to talk." if e.payload["muted"] else "Microphone on.")


async def keyboard(runtime: Runtime, voice: VoiceInput, screen: Screen) -> None:
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    keys: asyncio.Queue[bytes] = asyncio.Queue()

    def listen() -> None:
        tty.setcbreak(fd)
        loop.add_reader(fd, lambda: keys.put_nowait(os.read(fd, 1)))

    def unlisten() -> None:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    listen()
    try:
        while True:
            key = (await keys.get()).lower()
            if key == b" ":
                if voice.capturing:
                    await voice.finish()
                elif await voice.start() is None:
                    screen.line("Microphone is muted: unmute it to talk.")
            elif key == b"s":
                await voice.abort()
                await runtime.cancel()
            elif key == b"t":
                await voice.abort()
                unlisten()
                try:
                    text = await loop.run_in_executor(None, input, "You (typing): ")
                finally:
                    listen()
                runtime.submit_text(text)
            elif key in (b"q", b"\x04"):
                return
    finally:
        unlisten()


async def main() -> int:
    missing = [tool for tool in ("pw-record", "pw-play", "wpctl") if not shutil.which(tool)]
    if missing:
        print(f"Missing PipeWire tools: {', '.join(missing)}")
        return 1
    if not sys.stdin.isatty():
        print("Run Autum in a terminal.")
        return 1
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=3.0)) as client:
        problem = await check_ready(client)
        if problem:
            print(problem)
            return 1
        print("Checking the microphone…")
        if not await check_microphone():
            print(MIC_HELP)
            return 1
        print("Loading the voice model…")
        await preload(client)

        bus = EventBus(new_id())
        screen = Screen(bus)
        speech = SpeechPlayer(bus, OmniSpeech(client))
        runtime = Runtime(bus, OmniChat(client), speech, mode=Mode.PUSH_TO_TALK, system_prompt=system_prompt)
        voice = VoiceInput(bus, runtime, OmniTranscriber(client))
        mute = asyncio.create_task(MuteMonitor(bus).run())
        print("\nAutum is ready (provisional pipeline D: LFM3-Omni hears you and answers in its own voice).")
        print(KEYS + "\n")
        try:
            await keyboard(runtime, voice, screen)
        finally:
            mute.cancel()
            await voice.abort()
            await runtime.cancel()
            speech.stop()
            try:
                await asyncio.wait_for(runtime.wait_until_settled(), 3)
            except asyncio.TimeoutError:
                pass
    print("Bye.")
    return 0
