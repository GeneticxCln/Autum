"""Hold-to-talk key anywhere on the desktop (plan §3.1: hold to speak,
release to end the turn), through the XDG GlobalShortcuts portal.

The desktop asks the user once to confirm or change the key. Autum sees
only that key's press and release, nothing else typed. GLib runs in its
own thread; presses are handed to the asyncio loop.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from typing import Callable

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SHORTCUTS = "org.freedesktop.portal.GlobalShortcuts"
APP_ID = "autum"
SHORTCUT_ID = "push-to-talk"
PREFERRED_KEY = "Meta+Z"
BIND_TIMEOUT_S = 300  # the user may take a while in the desktop's dialog


def request_path(unique_name: str, token: str) -> str:
    """Where the portal will publish the Response for a request made with token."""
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"{PORTAL_PATH}/request/{sender}/{token}"


class KeyRouter:
    """Turns portal Activated/Deactivated signals into press/release calls for our key only."""

    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        self.session: str | None = None
        self.on_press = on_press
        self.on_release = on_release

    def signal(self, name: str, session: str, shortcut_id: str) -> None:
        if session != self.session or shortcut_id != SHORTCUT_ID:
            return
        if name == "Activated":
            self.on_press()
        elif name == "Deactivated":
            self.on_release()


class PushToTalkKey:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_status: Callable[[str], None],
    ) -> None:
        self.loop = loop
        self.on_status = on_status
        self._stopped = threading.Event()
        self._glib_loop = None
        self._glib_context = None
        self.router = KeyRouter(
            lambda: self._dispatch(on_press),
            lambda: self._dispatch(on_release),
        )

    def _dispatch(self, fn: Callable, *args) -> None:
        if self._stopped.is_set() or self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass

    def start(self) -> None:
        threading.Thread(target=self._run, name="autum-hotkey", daemon=True).start()

    def stop(self) -> None:
        """Stop the GLib portal loop cleanly on shutdown."""
        self._stopped.set()
        loop = self._glib_loop
        ctx = self._glib_context
        if loop is not None and ctx is not None:
            try:
                ctx.invoke(loop.quit)
            except Exception:
                pass

    def _status(self, message: str) -> None:
        self._dispatch(self.on_status, message)

    def _run(self) -> None:
        try:
            from gi.repository import Gio, GLib
        except ImportError:
            self._status("Hold-to-talk key unavailable (PyGObject missing); hold the bubble to talk.")
            return
        context = GLib.MainContext.new()
        context.push_thread_default()
        self._glib_context = context
        try:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            self._register(bus, Gio, GLib)
            session = self._create_session(bus, context, Gio, GLib)
            if self._stopped.is_set():
                return
            for name in ("Activated", "Deactivated"):
                bus.signal_subscribe(
                    PORTAL, SHORTCUTS, name, PORTAL_PATH, None, Gio.DBusSignalFlags.NONE,
                    lambda _c, _s, _p, _i, signal, params: self.router.signal(signal, *params.unpack()[:2]),
                )
            self.router.session = session
            key = self._bind(bus, context, session, Gio, GLib)
            if self._stopped.is_set():
                return
            self._status(f"Hold {key} anywhere to talk to Autum (or hold the bubble).")
            glib_loop = GLib.MainLoop.new(context, False)
            self._glib_loop = glib_loop
            if not self._stopped.is_set():
                glib_loop.run()
        except Exception as exc:  # the portal is optional: Autum still works with the bubble
            if not self._stopped.is_set():
                self._status(f"Hold-to-talk key unavailable ({exc}); hold the bubble to talk.")
        finally:
            self._glib_loop = None
            self._glib_context = None
            context.pop_thread_default()

    def _register(self, bus, Gio, GLib) -> None:
        """Tell the portal who we are, so the desktop remembers the key for Autum."""
        try:
            bus.call_sync(
                PORTAL, PORTAL_PATH, "org.freedesktop.host.portal.Registry", "Register",
                GLib.Variant("(sa{sv})", (APP_ID, {})), None, Gio.DBusCallFlags.NONE, 5000, None,
            )
        except GLib.Error:
            pass  # older portals: works without registration, the key may not be remembered

    def _create_session(self, bus, context, Gio, GLib) -> str:
        token = "autum_session_" + secrets.token_hex(4)
        results = self._request(
            bus, context, "CreateSession", (), "(a{sv})",
            {"session_handle_token": GLib.Variant("s", token)}, 10, Gio, GLib,
        )
        return str(results["session_handle"])

    def _bind(self, bus, context, session: str, Gio, GLib) -> str:
        shortcut = (SHORTCUT_ID, {
            "description": GLib.Variant("s", "Hold to talk to Autum"),
            "preferred_trigger": GLib.Variant("s", PREFERRED_KEY),
        })
        results = self._request(
            bus, context, "BindShortcuts", (session, [shortcut], ""), "(oa(sa{sv})sa{sv})",
            {}, BIND_TIMEOUT_S, Gio, GLib,
        )
        for shortcut_id, info in results.get("shortcuts", []):
            if shortcut_id == SHORTCUT_ID:
                return info.get("trigger_description") or "the Autum key"
        return "the Autum key"

    def _request(self, bus, context, method, args, signature, extra, timeout_s, Gio, GLib) -> dict:
        """Call a portal method and wait for its Response signal."""
        token = "autum_" + secrets.token_hex(6)
        path = request_path(bus.get_unique_name(), token)
        reply: dict = {}

        def on_response(_c, _s, _p, _i, _signal, params):
            reply["code"], reply["results"] = params.unpack()

        sub = bus.signal_subscribe(
            PORTAL, "org.freedesktop.portal.Request", "Response", path, None, Gio.DBusSignalFlags.NONE, on_response
        )
        try:
            options = {"handle_token": GLib.Variant("s", token), **extra}
            bus.call_sync(
                PORTAL, PORTAL_PATH, SHORTCUTS, method, GLib.Variant(signature, (*args, options)),
                GLib.VariantType("(o)"), Gio.DBusCallFlags.NONE, 10000, None,
            )
            deadline = time.monotonic() + timeout_s
            while "code" not in reply:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{method} got no answer")
                if not context.iteration(False):
                    time.sleep(0.02)
        finally:
            bus.signal_unsubscribe(sub)
        if reply["code"] != 0:
            raise RuntimeError(f"{method} was {'cancelled' if reply['code'] == 1 else 'refused'}")
        return reply["results"]
