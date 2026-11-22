"""Omni adapters against a fake Ollama (no network, no model)."""

import asyncio
import base64
import json

import httpx

from autum.ollama import OmniChat, OmniSpeech, OmniTranscriber, check_ready, to_openai


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
