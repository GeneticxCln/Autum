"""Autum releases the GPU on exit and survives a failed model load at startup."""

import asyncio
import json

import httpx

from autum.app import wait_for_startup_preload
from autum.ollama import release_models


def test_exit_unloads_only_autums_models_that_are_still_loaded():
    unloaded = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "lfm3-omni:2b"}, {"name": "gemma4:12b"}, {"name": "other:7b"}]})
        body = json.loads(request.content)
        assert body["keep_alive"] == 0
        unloaded.append(body["model"])
        return httpx.Response(200, json={})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await release_models(client, ["lfm3-omni:2b", "gemma4:12b", "qwen3-vl:8b", "gemma4:12b"])

    asyncio.run(go())
    # qwen3-vl was not loaded (unloading it would load it first); other:7b is not Autum's.
    assert sorted(unloaded) == ["gemma4:12b", "lfm3-omni:2b"]


def test_exit_unload_tolerates_ollama_being_down():
    def handler(request):
        raise httpx.ConnectError("down")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await release_models(client, ["lfm3-omni:2b"])

    asyncio.run(go())  # no exception


def test_failed_startup_model_load_is_reported_not_raised():
    async def failing_load():
        raise RuntimeError("Ollama preload failed for gemma4:12b (500): out of memory")

    async def go():
        return await wait_for_startup_preload(failing_load(), asyncio.Event())

    stopped, error = asyncio.run(go())
    assert stopped is False
    assert "out of memory" in str(error)


def test_stop_during_startup_load_cancels_it():
    async def go():
        stop = asyncio.Event()
        started = asyncio.Event()

        async def slow_load():
            started.set()
            await asyncio.sleep(30)

        task = asyncio.ensure_future(wait_for_startup_preload(slow_load(), stop))
        await started.wait()
        stop.set()
        return await asyncio.wait_for(task, 2)

    assert asyncio.run(go()) == (True, None)


def test_successful_startup_load():
    async def ok():
        return None

    assert asyncio.run(wait_for_startup_preload(ok(), asyncio.Event())) == (False, None)
