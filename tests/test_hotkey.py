"""Hold-to-talk key: request paths and press/release routing (no D-Bus needed)."""

from autum.hotkey import SHORTCUT_ID, KeyRouter, request_path


def test_request_path_follows_the_portal_rule():
    assert request_path(":1.533", "autum_ab") == "/org/freedesktop/portal/desktop/request/1_533/autum_ab"


def test_only_our_key_in_our_session_presses_and_releases():
    calls = []
    router = KeyRouter(lambda: calls.append("press"), lambda: calls.append("release"))
    router.session = "/session/ours"
    router.signal("Activated", "/session/ours", SHORTCUT_ID)
    router.signal("Activated", "/session/other", SHORTCUT_ID)  # another app's session
    router.signal("Activated", "/session/ours", "something-else")
    router.signal("Deactivated", "/session/ours", SHORTCUT_ID)
    assert calls == ["press", "release"]
