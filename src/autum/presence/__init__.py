"""Autum's visual presence (WP4): a Qt Quick window fed by the event socket."""

from pathlib import Path


def has_layer_shell() -> bool:
    """True when Qt's Wayland wlr-layer-shell plugin and QML module are installed."""
    try:
        from PySide6.QtCore import QLibraryInfo
    except ImportError:
        return False
    qml = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.QmlImportsPath)) / "org" / "kde" / "layershell" / "qmldir"
    plugin = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "wayland-shell-integration" / "liblayer-shell.so"
    return qml.is_file() and plugin.is_file()
