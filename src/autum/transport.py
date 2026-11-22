"""Event socket for out-of-process components (ADR 0001, contract §8).

Clients such as the visual presence connect to a Unix socket that only
the user's own account can open. They receive every event as a JSON line,
starting with a snapshot of the current state, and may send a small set
of user commands back.

Memory stays bounded: each client has a fixed-size queue, and a client
too slow to keep up is disconnected. It reconnects and gets a fresh
snapshot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from autum.bus import EventBus
from autum.events import Event

log = logging.getLogger("autum.transport")

CLIENT_QUEUE = 512
MAX_LINE = 8192
MAX_TEXT = 2000
COMMANDS = {"talk_start", "talk_finish", "stop", "text", "set_mode", "approve", "deny"}
MODES = {"push_to_talk", "hands_free"}
SNAPSHOT_TYPES = ("runtime.state_changed", "audio.mute_changed")

CommandHandler = Callable[[dict], Awaitable[None]]


def default_socket_path() -> Path:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        raise RuntimeError("XDG_RUNTIME_DIR is not set; cannot create a private socket")
    return Path(runtime_dir) / "autum" / "events.sock"


def parse_command(line: bytes) -> dict | None:
    """Validate one command line from a client; None if it is not acceptable."""
    try:
        data = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("command") not in COMMANDS:
        return None
    if data["command"] == "text":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            return None
        return {"command": "text", "text": text}
    if data["command"] == "set_mode":
        mode = data.get("mode")
        if mode not in MODES or set(data) != {"command", "mode"}:
            return None
        return {"command": "set_mode", "mode": mode}
    if set(data) != {"command"}:
        return None
    return {"command": data["command"]}


class EventServer:
    def __init__(self, bus: EventBus, on_command: CommandHandler, path: Path | None = None) -> None:
        self.bus = bus
        self.on_command = on_command
        self.path = path or default_socket_path()
        self._snapshot: dict[str, Event] = {}
        self._clients: set[asyncio.Queue] = set()
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task] = set()
        bus.subscribe(self._publish)

    async def start(self) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if self.path.exists():
            if await self._someone_listening():
                raise RuntimeError(f"another Autum is already running ({self.path})")
            self.path.unlink()  # stale socket from a crashed run
        old_umask = os.umask(0o177)  # socket file created as 0600
        try:
            self._server = await asyncio.start_unix_server(self._serve, path=str(self.path), limit=MAX_LINE)
        finally:
            os.umask(old_umask)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in list(self._tasks):
            task.cancel()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def _someone_listening(self) -> bool:
        try:
            _, writer = await asyncio.open_unix_connection(str(self.path))
        except OSError:
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    def _publish(self, event: Event) -> None:
        if event.type in SNAPSHOT_TYPES:
            self._snapshot[event.type] = event
        line = event.to_json().encode() + b"\n"
        for queue in list(self._clients):
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                self._clients.discard(queue)
                _drop_slow(queue)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        queue: asyncio.Queue = asyncio.Queue(CLIENT_QUEUE)
        for event in sorted(self._snapshot.values(), key=lambda e: e.mono_ns):  # replay in time order
            queue.put_nowait(event.to_json().encode() + b"\n")
        self._clients.add(queue)
        sender = asyncio.create_task(self._send(queue, writer))
        self._tasks.add(sender)
        sender.add_done_callback(self._tasks.discard)
        try:
            while not sender.done():
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    break  # a line longer than MAX_LINE: drop this client
                if not line:
                    break
                command = parse_command(line)
                if command is None:
                    log.warning("ignored invalid command from a client")
                    continue
                try:
                    await self.on_command(command)
                except Exception:
                    log.exception("command %s failed", command["command"])
        except ConnectionError:
            pass
        finally:
            self._clients.discard(queue)
            sender.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _send(self, queue: asyncio.Queue, writer: asyncio.StreamWriter) -> None:
        try:
            while (line := await queue.get()) is not None:
                writer.write(line)
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()  # ends the reader too, so the client reconnects


def _drop_slow(queue: asyncio.Queue) -> None:
    """Stop a client that cannot keep up: clear its backlog and end its sender."""
    while not queue.empty():
        queue.get_nowait()
    queue.put_nowait(None)
