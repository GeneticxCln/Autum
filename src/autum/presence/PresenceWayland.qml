// Wayland wrapper around Presence.qml using wlr-layer-shell (layer-shell-qt).
// Keeps the particle cloud anchored to the bottom-right corner across every
// virtual desktop, activity, and normal/fullscreen window without stealing
// focus or appearing on the taskbar. Loaded only on Wayland when
// org.kde.layershell is available.
import QtQuick
import org.kde.layershell as LayerShell

Presence {
    flags: Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus
    LayerShell.Window.anchors: LayerShell.Window.AnchorBottom | LayerShell.Window.AnchorRight
    LayerShell.Window.layer: LayerShell.Window.LayerOverlay
    LayerShell.Window.exclusionZone: 0
    LayerShell.Window.keyboardInteractivity: LayerShell.Window.KeyboardInteractivityNone
    LayerShell.Window.activateOnShow: false
    LayerShell.Window.wantsToBeOnActiveScreen: true
    LayerShell.Window.scope: "autum"
    LayerShell.Window.margins.right: margin - pad
    LayerShell.Window.margins.bottom: margin - pad
}
