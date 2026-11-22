"""Omni adapters against a fake Ollama (no network, no model)."""

import asyncio
import base64
import json

import httpx

from autum.ollama import OmniChat, OmniSpeech, OmniTranscriber, check_ready, to_openai
from autum.tools import ToolCall


def run(coro):
    return asyncio.run(coro)


def fake_ollama(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_audio_turns_become_input_audio_and_untranscribed_old_turns_are_marked():
    assert to_openai({"role": "user", "content": "", "audio": b"WAV", "turn_id": "t"}) == {
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(b"WAV").decode(), "format": "wav"}}],
    }
    assert to_openai({"role": "user", "content": "", "turn_id": "t"})["content"] == "(unclear speech)"
    assert to_openai({"role": "assistant", "content": "Hi."}) == {"role": "assistant", "content": "Hi."}
    assert to_openai({"role": "assistant", "content": "Sure. [interrupted by the user]"}) == {
        "role": "assistant",
        "content": "Sure.",
    }


def test_chat_streams_deltas_with_pinned_sampling():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        chunks = [{"choices": [{"delta": {"content": c}}]} for c in ("Hel", "lo.")]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    async def scenario():
        async with fake_ollama(handler) as client:
            return [t async for t in OmniChat(client).stream([{"role": "user", "content": "hi"}], operation_id="op")]

    assert run(scenario()) == ["Hel", "lo."]
    assert seen["temperature"] == 0 and seen["seed"] == 42 and seen["model"] == "lfm3-omni:2b"


def test_errors_are_reported_not_swallowed():
    async def scenario():
        async with fake_ollama(lambda r: httpx.Response(500, text="model crashed")) as client:
            try:
                async for _ in OmniSpeech(client).stream("hi"):
                    pass
            except RuntimeError as exc:
                return str(exc)

    assert "500" in run(scenario())


def test_transcription_and_readiness():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        return httpx.Response(200, json={"text": " What time is it? "})

    async def scenario():
        async with fake_ollama(handler) as client:
            return await OmniTranscriber(client).transcribe(b"WAV"), await check_ready(client)

    text, problem = run(scenario())
    assert text == "What time is it?"
    assert "does not have lfm3-omni:2b" in problem


def test_chat_streams_tool_calls_and_formats_tool_messages():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "set_timer", "arguments": "{\"seconds\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "60,\"label\":\"tea\"}"}}]}}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    async def scenario():
        async with fake_ollama(handler) as client:
            tools = [{"type": "function", "function": {"name": "set_timer", "parameters": {"type": "object"}}}]
            return [t async for t in OmniChat(client).stream([{"role": "user", "content": "set timer"}], operation_id="op", tools=tools)]

    out = run(scenario())
    assert out == [ToolCall(id="call_1", name="set_timer", arguments={"seconds": 60, "label": "tea"})]
    assert seen["tools"][0]["function"]["name"] == "set_timer"
    assert to_openai({"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'}) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok":true}',
    }


def test_speech_falls_back_to_lfm3_omni_when_hf_gguf_lacks_speech_capability():
    from autum.ollama import MODEL, resolve_speech_model

    calls = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/api/show":
            caps = ["completion", "audio"] if "hf.co/" in body.get("name", "") else ["completion", "audio", "speech"]
            return httpx.Response(200, json={"capabilities": caps})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/v1/audio/speech":
            calls.append(body["model"])
            if "hf.co/" in body["model"]:
                return httpx.Response(400, json={"error": {"message": f"{body['model']} does not support speech"}})
            return httpx.Response(200, content=b"\x01\x02\x03\x04")
        return httpx.Response(404)

    async def scenario():
        async with fake_ollama(handler) as client:
            resolved = await resolve_speech_model(client, "hf.co/NeosQ/LFM3-Omni-2B-Uncensored:F16")
            synth = OmniSpeech(client, model="hf.co/NeosQ/LFM3-Omni-2B-Uncensored:F16")
            pcm = b"".join([c async for c in synth.stream("Hello")])
            return resolved, synth.model, pcm

    resolved, active_model, pcm = run(scenario())
    assert resolved == MODEL
    assert active_model == MODEL
    assert pcm == b"\x01\x02\x03\x04"
    assert calls == ["hf.co/NeosQ/LFM3-Omni-2B-Uncensored:F16", MODEL]
