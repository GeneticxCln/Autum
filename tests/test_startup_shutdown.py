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


def test_release_models_and_health_checks_match_untagged_latest_and_live_voice(monkeypatch):
    from autum import diagnostics
    from autum.config import Config

    unloaded = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "gemma4:latest"}, {"name": "lfm3-omni:2b"}]})
        body = json.loads(request.content)
        assert body["keep_alive"] == 0
        unloaded.append(body["model"])
        return httpx.Response(200, json={})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await release_models(client, ["gemma4", "gemma4:latest", "lfm3-omni:2b"])

    asyncio.run(go())
    assert sorted(unloaded) == ["gemma4:latest", "lfm3-omni:2b"]

    # Self-diagnosis matches untagged "gemma4" -> "gemma4:latest" and uses live_state["voice_model"]
    monkeypatch.setattr(
        diagnostics,
        "query_ollama_models",
        lambda _url, timeout_s=2.0: [
            {"name": "gemma4:latest", "size_gb": 8.0, "loaded": True, "status": "100% GPU"},
            {"name": "qwen3-vl:latest", "size_gb": 5.0, "loaded": False, "status": ""},
            {"name": "big-f16-omni:F16", "size_gb": 5.0, "loaded": False, "status": ""},
            {"name": "lfm3-omni:2b", "size_gb": 1.4, "loaded": True, "status": "100% GPU"},
        ],
    )
    monkeypatch.setattr(diagnostics, "gpu_vram_gb", lambda: 12.0)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(diagnostics, "_run_cmd", lambda cmd, timeout_s=3.0: (0, "Volume: 0.50"))

    cfg = Config(model="big-f16-omni:F16", chat_model="gemma4", vision_model="qwen3-vl")
    rep = diagnostics.run_system_health_checks(cfg, live_state={"voice_model": "lfm3-omni:2b"})
    models_check = next(c for c in rep["checks"] if c["section"] == "models")
    # 8.0 + 1.4 + 2.5 = 11.9 <= 12.0 GB (would falsely warn as 8.0 + 5.0 + 2.5 = 15.5 GB without live voice_model)
    assert models_check["status"] == "ok"
    assert "Voice=lfm3-omni:2b" in models_check["summary"]
