"""Run the visual presence: python3 -m autum.presence [--socket PATH] [--measure N]

Normally started by `python3 -m autum`. With --measure N it exits after
N state changes have reached the screen and prints the lag figures.
"""

import argparse
import ctypes
import ctypes.util
import sys
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QRegion
from PySide6.QtQml import QQmlApplicationEngine

from autum.presence import has_layer_shell
from autum.presence.model import PresenceModel
from autum.transport import default_socket_path


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 -m autum.presence")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--measure", type=int, default=0)
    args = parser.parse_args()

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Autum")
    model = PresenceModel(args.socket or default_socket_path(), measure=args.measure)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("presence", model)
    qml_name = "PresenceWayland.qml" if app.platformName().startswith("wayland") and has_layer_shell() else "Presence.qml"
    engine.load(QUrl.fromLocalFile(str(Path(__file__).with_name(qml_name))))
    if not engine.rootObjects():
        return 1
    window = engine.rootObjects()[0]
    window.frameSwapped.connect(model.frame_swapped)
    window.show()
    # Only the round orb takes clicks; the transparent corners pass them through.
    window.setMask(QRegion(0, 0, window.width(), window.height(), QRegion.RegionType.Ellipse))
    if app.platformName() == "xcb":
        QTimer.singleShot(300, lambda: skip_taskbar(int(window.winId())))
    model.start()
    return app.exec()


def skip_taskbar(window_id: int) -> None:
    """Ask an X11 window manager to keep the bubble off the taskbar and on all desktops.

    Standard EWMH requests for a window that is already shown, sent through
    the system's libX11; Qt has no API for them.
    """
    try:
        x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
    except (OSError, TypeError):
        return
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.c_void_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        return

    class ClientMessage(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
            ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
            ("format", ctypes.c_int), ("data", ctypes.c_long * 5), ("_pad", ctypes.c_long * 12),
        ]

    atom = lambda name: x11.XInternAtom(display, name, 0)  # noqa: E731
    root = x11.XDefaultRootWindow(display)
    mask = (1 << 20) | (1 << 19)  # SubstructureRedirect | SubstructureNotify
    for message_type, data in (
        (atom(b"_NET_WM_STATE"), (1, atom(b"_NET_WM_STATE_SKIP_TASKBAR"), atom(b"_NET_WM_STATE_SKIP_PAGER"), 1, 0)),
        (atom(b"_NET_WM_STATE"), (1, atom(b"_NET_WM_STATE_STICKY"), 0, 1, 0)),
        (atom(b"_NET_WM_DESKTOP"), (0xFFFFFFFF, 1, 0, 0, 0)),
    ):
        event = ClientMessage(
            type=33,  # ClientMessage
            send_event=1, display=display, window=window_id,
            message_type=message_type, format=32,
            data=(ctypes.c_long * 5)(*data),
        )
        x11.XSendEvent(display, root, 0, mask, ctypes.byref(event))
    x11.XFlush(display)
    x11.XCloseDisplay(display)


if __name__ == "__main__":
    sys.exit(main())
