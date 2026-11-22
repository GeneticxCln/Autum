// Autum's presence (WP4): a 3D stippled face suspended inside a frosted
// stardust orb in the screen corner, nothing else.
// Every visual follows `presence`, which only reflects runtime events.
//
// The state changes how the face and surrounding stardust shell behave:
//   idle       pure silver-ice, slow orbit, gentle breathing head sway
//   listening  cool ice-cyan, gathered inward, attentive forward gaze
//   thinking   silver-violet, swirling stardust bands, subtle crown wave
//   speaking   luminous silver-blue, articulated jaw + outward stardust burst
// Indicators are part of the orb: a warm edge aura while the mic is open,
// muted grey with a lower-right gap while muted, and a cool cyan-teal outer
// halo + attentive eye sheen while observing an image, document, window, or screen.
import QtQuick
import QtQuick.Window
import Autum 1.0

Window {
    id: win
    // Keep the physical pixel size consistent between unscaled X11/XWayland
    // and native Wayland HiDPI (e.g. 1.5x scale on 4K).
    readonly property real px: 1 / Math.max(1, Screen.devicePixelRatio)
    readonly property int size: Math.round(300 * px)   // orb diameter at rest, in physical pixels
    readonly property int pad: Math.round(size * 0.25) // room for voice burst & edge auras
    readonly property int margin: Math.round(24 * px)
    width: size + 2 * pad
    height: width
    x: Screen.virtualX + Screen.desktopAvailableWidth - width - margin + pad
    y: Screen.virtualY + Screen.desktopAvailableHeight - height - margin + pad
    visible: false  // shown by __main__.py
    title: "Autum"
    color: "transparent"
    flags: Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus

    readonly property string st: presence.connected ? presence.state : "offline"
    readonly property bool busy: st === "processing" || st === "thinking" || st === "loading"
    readonly property bool hearing: st === "listening" || st === "speech_detected"
    readonly property bool speaking: st === "speaking"
    readonly property bool micOpen: presence.connected && presence.microphone === "open"
    readonly property bool muted: presence.connected && presence.microphone === "muted"
    readonly property bool observing: presence.connected && presence.observation !== "none"
    readonly property real radius: size / 2

    // Frosted silver-ice palette anchored to the holographic face artwork;
    // subtle, low-glare state tints keep every runtime state immediately legible.
    readonly property var stateColors: ({
        "idle": "#dce6f2",
        "loading": "#9aa8bc",
        "listening": "#bce8f2",
        "speech_detected": "#b4f0dc",
        "processing": "#c9c2f0",
        "thinking": "#c9c2f0",
        "speaking": "#d8ecff",
        "interrupted": "#e6bfa0",
        "waiting_for_approval": "#ebd39b",
        "error": "#e094a0",
        "offline": "#6a7385"
    })
    property color tint: muted ? "#8c93a3" : presence.error !== "" ? "#e094a0" : (stateColors[st] || "#dce6f2")
    Behavior on tint { ColorAnimation { duration: 300 } }

    // Motion parameters; each eases smoothly when the state changes.
    property real loose: hearing ? 0.04 : busy ? 0.08 : speaking ? 0.06 : st === "offline" ? 0.04 : 0.12
    property real spinSpeed: busy ? 1.5 : hearing ? 0.42 : st === "offline" ? 0.05 : 0.20   // radians per second
    property real swirlSpeed: busy ? 2.1 : 0.0
    property real gather: hearing ? 0.92 : st === "interrupted" ? 0.86 : 1.0
    property real brightness: st === "offline" ? 0.48 : muted ? 0.74 : hearing || speaking ? 1.0 : 0.94
    property real warmth: micOpen ? 0.65 : 0.0
    property real observeGlow: observing ? 0.75 : 0.0
    property real gap: muted ? 1.0 : 0.0
    Behavior on loose { NumberAnimation { duration: 600; easing.type: Easing.InOutQuad } }
    Behavior on spinSpeed { NumberAnimation { duration: 800; easing.type: Easing.InOutQuad } }
    Behavior on swirlSpeed { NumberAnimation { duration: 800; easing.type: Easing.InOutQuad } }
    Behavior on gather { NumberAnimation { duration: 350; easing.type: Easing.OutBack } }
    Behavior on brightness { NumberAnimation { duration: 400 } }
    Behavior on warmth { NumberAnimation { duration: 250 } }
    Behavior on observeGlow { NumberAnimation { duration: 300 } }
    Behavior on gap { NumberAnimation { duration: 400 } }

    property real voice: presence.level
    Behavior on voice { SmoothedAnimation { velocity: 3.5 } }

    // Time-driven phases updated at 30 Hz.
    property real spin: 0
    property real swirl: 0
    property real phase: 0
    property real lastTick: Date.now()
    Timer {
        interval: 33
        running: true
        repeat: true
        onTriggered: {
            const now = Date.now()
            const dt = Math.min((now - win.lastTick) / 1000, 0.1)
            win.lastTick = now
            win.spin += win.spinSpeed * dt
            win.swirl += win.swirlSpeed * dt
            win.phase += dt
            faceCloud.redraw()
        }
    }

    FaceCloud {
        id: faceCloud
        anchors.fill: parent
        orbRadius: win.radius
        tint: win.tint
        voice: win.voice
        spin: win.spin
        swirl: win.swirl
        phase: win.phase
        gather: win.gather
        loose: win.loose
        brightness: win.brightness
        gap: win.gap
        warmth: win.warmth
        observeGlow: win.observeGlow
        speaking: win.speaking
        hearing: win.hearing
        busy: win.busy
        observing: win.observing
        lookX: talkArea.lookX
        lookY: talkArea.lookY
        hovering: talkArea.hoveringOrb
    }

    // Hold the orb to talk; hovering lets the 3D head turn toward your cursor.
    MouseArea {
        id: talkArea
        anchors.fill: parent
        hoverEnabled: true
        property real lookX: 0
        property real lookY: 0
        property bool hoveringOrb: false
        onPositionChanged: (mouse) => {
            const dx = (mouse.x - width / 2) / win.radius
            const dy = (mouse.y - height / 2) / win.radius
            hoveringOrb = Math.sqrt(dx * dx + dy * dy) <= 1.15
            if (hoveringOrb) {
                lookX = Math.max(-1, Math.min(1, dx))
                lookY = Math.max(-1, Math.min(1, dy))
            }
        }
        onExited: hoveringOrb = false
        onPressed: (mouse) => {
            const dx = mouse.x - width / 2, dy = mouse.y - height / 2
            if (Math.sqrt(dx * dx + dy * dy) > win.radius * 1.05) { mouse.accepted = false; return }
            presence.talkStart()
        }
        onReleased: presence.talkFinish()
        onCanceled: presence.talkFinish()
    }
}
