"""Event socket: snapshot, ordering, commands, permissions, bounded memory."""

import asyncio
import json
import os
import stat

import pytest

from autum.bus import EventBus
from autum.events import Event
from autum.transport import CLIENT_QUEUE, EventServer, parse_command


def run(coro):
    return asyncio.run(coro)


async def read_events(reader, n, timeout=2.0):
    out = []
    for _ in range(n):
        line = await asyncio.wait_for(reader.readline(), timeout)
        out.append(Event.from_json(line))
    return out


def test_commands_are_validated():
    assert parse_command(b'{"command": "talk_start"}') == {"command": "talk_start"}
    assert parse_command(b'{"command": "text", "text": "hello"}') == {"command": "text", "text": "hello"}
    assert parse_command(b'{"command": "set_mode", "mode": "hands_free"}') == {"command": "set_mode", "mode": "hands_free"}
    assert parse_command(b'{"command": "set_mode", "mode": "push_to_talk"}') == {"command": "set_mode", "mode": "push_to_talk"}
    for bad in [b"not json", b"[1]", b'{"command": "rm -rf"}', b'{"command": "stop", "extra": 1}',
                b'{"command": "text", "text": ""}', b'{"command": "text", "text": 5}',
                b'{"command": "set_mode", "mode": "wake_word"}', b'{"command": "set_mode"}',
                b'{"command": "text", "text": "' + b"x" * 2001 + b'"}']:
        assert parse_command(bad) is None, bad


def test_client_gets_a_snapshot_then_live_events_in_order(tmp_path):
    async def scenario():
        bus = EventBus("s")
        server = EventServer(bus, on_command=lambda c: asyncio.sleep(0), path=tmp_path / "a" / "events.sock")
        await server.start()
        bus.emit("audio.mute_changed", {"muted": False})
        bus.emit("runtime.state_changed", {"from": "idle", "to": "listening", "reason": "x", "microphone": "open", "observation": "none"})
        reader, writer = await asyncio.open_unix_connection(str(server.path))
        snapshot = await read_events(reader, 2)
        await asyncio.sleep(0.01)
        bus.emit("audio.capture_started")
        bus.emit("audio.capture_stopped")
        live = await read_events(reader, 2)
        mode_dir = stat.S_IMODE(os.stat(server.path.parent).st_mode)
        mode_sock = stat.S_IMODE(os.stat(server.path).st_mode)
        writer.close()
        await server.close()
        return snapshot, live, mode_dir, mode_sock, server.path.exists()

    snapshot, live, mode_dir, mode_sock, still_there = run(scenario())
    assert [e.type for e in snapshot] == ["audio.mute_changed", "runtime.state_changed"]  # as they happened
    assert snapshot[1].payload["to"] == "listening"
    assert [e.type for e in live] == ["audio.capture_started", "audio.capture_stopped"]
    assert mode_dir == 0o700 and mode_sock == 0o600
    assert not still_there  # removed on close


def test_valid_commands_reach_the_handler_and_invalid_ones_do_not(tmp_path):
    async def scenario():
        got = []

        async def handler(command):
            got.append(command)

        server = EventServer(EventBus("s"), handler, path=tmp_path / "events.sock")
        await server.start()
        reader, writer = await asyncio.open_unix_connection(str(server.path))
        writer.write(b'{"command": "talk_start"}\nnonsense\n{"command": "text", "text": "hi"}\n')
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()
        await server.close()
        return got

    assert run(scenario()) == [{"command": "talk_start"}, {"command": "text", "text": "hi"}]


def test_a_client_that_stops_reading_is_dropped_not_buffered_forever(tmp_path):
    async def scenario():
        bus = EventBus("s")
        server = EventServer(bus, lambda c: asyncio.sleep(0), path=tmp_path / "events.sock")
        await server.start()
        reader, writer = await asyncio.open_unix_connection(str(server.path))
        await asyncio.sleep(0.01)
        assert server.client_count == 1
        for _ in range(CLIENT_QUEUE * 40):  # far more than the socket and queue can hold
            bus.emit("audio.capture_started")
        await asyncio.sleep(0.1)
        count = server.client_count
        writer.close()
        await server.close()
        return count

    assert run(scenario()) == 0


def test_a_second_server_refuses_and_a_stale_socket_is_replaced(tmp_path):
    async def scenario():
        path = tmp_path / "events.sock"
        first = EventServer(EventBus("s"), lambda c: asyncio.sleep(0), path=path)
        await first.start()
        with pytest.raises(RuntimeError, match="already running"):
            await EventServer(EventBus("s2"), lambda c: asyncio.sleep(0), path=path).start()
        first._server.close()  # simulate a crash: socket file left behind
        await first._server.wait_closed()
        replacement = EventServer(EventBus("s3"), lambda c: asyncio.sleep(0), path=path)
        await replacement.start()
        await replacement.close()

    run(scenario())
