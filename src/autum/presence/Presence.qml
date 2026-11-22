// Autum's presence (WP4): a particle cloud in the screen corner, nothing else.
// Every visual follows `presence`, which only reflects runtime events.
//
// About 260 soft points sit on a sphere that turns slowly in 3D; nearer
// points are larger and brighter. The state changes how they move:
//   idle      loose, drifting, slow turn
//   listening gathered tight
//   thinking  orbiting in swirling bands
//   speaking  pushed outward with Autum's voice
// The microphone is part of the cloud: a warm rim while it is open, grey
// with a gap while it is muted, nothing extra while it is off.
import QtQuick
import QtQuick.Window
import QtQuick.Shapes

Window {
    id: win
    // Keep the physical pixel size consistent between unscaled X11/XWayland
    // and native Wayland HiDPI (e.g. 1.5x scale on 4K).
    readonly property real px: 1 / Math.max(1, Screen.devicePixelRatio)
    readonly property int size: Math.round(300 * px)   // cloud diameter at rest, in physical pixels
    readonly property int pad: Math.round(size * 0.25) // room for the voice burst
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

    // Muted, low-glare colours per state.
    readonly property var stateColors: ({
        "idle": "#8fa3d9", "loading": "#8a90a8", "listening": "#7fc4cf", "speech_detected": "#86cdb8",
        "processing": "#a996d6", "thinking": "#a996d6", "speaking": "#8ab8dc", "interrupted": "#d0a07a",
        "waiting_for_approval": "#d1b47a", "error": "#c9808c", "offline": "#5c6178"
    })
    property color tint: muted ? "#7c8091" : presence.error !== "" ? "#c9808c" : (stateColors[st] || "#8fa3d9")
    Behavior on tint { ColorAnimation { duration: 300 } }

    // Motion parameters; each eases smoothly when the state changes.
    property real loose: hearing ? 0.03 : busy ? 0.08 : speaking ? 0.06 : st === "offline" ? 0.05 : 0.16
    property real spinSpeed: busy ? 1.6 : hearing ? 0.45 : st === "offline" ? 0.05 : 0.22   // radians per second
    property real swirlSpeed: busy ? 2.2 : 0.0
    property real gather: hearing ? 0.86 : st === "interrupted" ? 0.8 : 1.0
    property real brightness: st === "offline" ? 0.45 : hearing || speaking ? 1.0 : 0.82
    property real warmth: micOpen ? 0.55 : 0.0
    property real observeGlow: observing ? 0.6 : 0.0
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

    // Time-driven phases. Points are laid out 30 times a second in one loop,
    // which keeps the cloud cheap enough to run all day.
    property real spin: 0
    property real swirl: 0
    property real phase: 0
    property real lastTick: Date.now()
    Timer {
        interval: 33  // 30 updates a second; the screen redraws only when points move
        running: true
        repeat: true
        onTriggered: {
            const now = Date.now()
            const dt = Math.min((now - win.lastTick) / 1000, 0.1)
            win.lastTick = now
            win.spin += win.spinSpeed * dt
            win.swirl += win.swirlSpeed * dt
            win.phase += dt
            win.layoutDots()
        }
    }

    function layoutDots() {
        const cx = width / 2, cy = height / 2
        for (let i = 0; i < dots.count; ++i) {
            const d = dots.itemAt(i)
            if (!d) continue
            const a = d.lon + spin + swirl * (0.4 + d.seedA) * (1 - Math.abs(d.by))
            const rx = d.ring * Math.cos(a), rz = d.ring * Math.sin(a)
            const ty = d.by * cosTilt - rz * sinTilt
            const depth = (d.by * sinTilt + rz * cosTilt + 1) / 2
            const dist = gather * (1
                + loose * Math.sin(phase * (0.6 + d.seedB) + d.seedC * 6.283)
                + voice * (0.15 + 0.45 * d.seedA))
            let visible = 1
            if (gap > 0) {  // the muted gap: points facing the lower right fade out
                const f = Math.atan2(ty, rx)
                visible = 1 - gap * Math.min(1, 2 * Math.max(0, 1 - Math.abs(f - 0.785) / 0.6))
            }
            d.x = cx + radius * dist * rx - d.width / 2
            d.y = cy + radius * dist * ty - d.height / 2
            d.z = depth
            d.scale = 0.45 + 0.75 * depth + 0.25 * d.seedB
            d.opacity = brightness * (0.28 + 0.72 * depth) * visible
        }
    }

    readonly property real tilt: 0.38
    readonly property real cosTilt: Math.cos(tilt)
    readonly property real sinTilt: Math.sin(tilt)
    readonly property real radius: size / 2

    // A faint dark backing keeps the points readable on light backgrounds.
    Shape {
        anchors.centerIn: parent
        width: win.size * 1.2; height: width
        preferredRendererType: Shape.CurveRenderer
        opacity: win.brightness
        ShapePath {
            strokeColor: "transparent"
            fillGradient: RadialGradient {
                centerX: win.size * 0.6; centerY: centerX; centerRadius: centerX
                focalX: centerX; focalY: centerY
                GradientStop { position: 0.0; color: Qt.rgba(0.07, 0.08, 0.12, 0.24) }
                GradientStop { position: 0.75; color: Qt.rgba(0.07, 0.08, 0.12, 0.12) }
                GradientStop { position: 1.0; color: Qt.rgba(0.07, 0.08, 0.12, 0.0) }
            }
            PathAngleArc { centerX: win.size * 0.6; centerY: win.size * 0.6; radiusX: win.size * 0.6; radiusY: win.size * 0.6; startAngle: 0; sweepAngle: 360 }
        }
    }

    Item {
        id: cloud
        anchors.fill: parent

        Repeater {
            id: dots
            model: 260
            Rectangle {
                // Fixed per-point values: position on a Fibonacci sphere plus seeds.
                readonly property real by: 1 - 2 * (index + 0.5) / 260
                readonly property real ring: Math.sqrt(1 - by * by)
                readonly property real lon: index * 2.399963
                readonly property real seedA: fract(Math.sin(index * 12.9898) * 43758.5453)
                readonly property real seedB: fract(Math.sin(index * 78.233) * 12543.123)
                readonly property real seedC: fract(Math.sin(index * 39.425) * 24634.633)
                function fract(v) { return v - Math.floor(v) }

                width: 6 * win.px; height: width; radius: width / 2
                color: win.tint
                opacity: 0

                // Soft halo around the point.
                Rectangle {
                    anchors.centerIn: parent
                    width: 17 * win.px; height: width; radius: width / 2
                    color: parent.color
                    opacity: 0.16
                }
            }
        }
    }

    // Microphone open: a warm glow along the cloud's edge.
    Shape {
        anchors.centerIn: parent
        width: win.size * 1.3; height: width
        preferredRendererType: Shape.CurveRenderer
        opacity: win.warmth
        ShapePath {
            strokeColor: "transparent"
            fillGradient: RadialGradient {
                centerX: win.size * 0.65; centerY: centerX; centerRadius: centerX
                focalX: centerX; focalY: centerY
                GradientStop { position: 0.0; color: Qt.rgba(0.91, 0.64, 0.48, 0.0) }
                GradientStop { position: 0.6; color: Qt.rgba(0.91, 0.64, 0.48, 0.0) }
                GradientStop { position: 0.74; color: Qt.rgba(0.91, 0.64, 0.48, 0.32) }
                GradientStop { position: 1.0; color: Qt.rgba(0.91, 0.64, 0.48, 0.0) }
            }
            PathAngleArc { centerX: win.size * 0.65; centerY: win.size * 0.65; radiusX: win.size * 0.65; radiusY: win.size * 0.65; startAngle: 0; sweepAngle: 360 }
        }
    }

    // Observation active (§3.2 / §5.4): a cool cyan-teal outer edge ring while
    // observing a screenshot, window, image, or document; invisible otherwise.
    Shape {
        anchors.centerIn: parent
        width: win.size * 1.38; height: width
        preferredRendererType: Shape.CurveRenderer
        opacity: win.observeGlow
        ShapePath {
            strokeColor: "transparent"
            fillGradient: RadialGradient {
                centerX: win.size * 0.69; centerY: centerX; centerRadius: centerX
                focalX: centerX; focalY: centerY
                GradientStop { position: 0.0; color: Qt.rgba(0.45, 0.85, 0.92, 0.0) }
                GradientStop { position: 0.68; color: Qt.rgba(0.45, 0.85, 0.92, 0.0) }
                GradientStop { position: 0.82; color: Qt.rgba(0.45, 0.85, 0.92, 0.34) }
                GradientStop { position: 1.0; color: Qt.rgba(0.45, 0.85, 0.92, 0.0) }
            }
            PathAngleArc { centerX: win.size * 0.69; centerY: win.size * 0.69; radiusX: win.size * 0.69; radiusY: win.size * 0.69; startAngle: 0; sweepAngle: 360 }
        }
    }

    // Hold the cloud to talk.
    MouseArea {
        anchors.fill: parent
        onPressed: (mouse) => {
            const dx = mouse.x - width / 2, dy = mouse.y - height / 2
            if (Math.sqrt(dx * dx + dy * dy) > win.radius * 1.05) { mouse.accepted = false; return }
            presence.talkStart()
        }
        onReleased: presence.talkFinish()
        onCanceled: presence.talkFinish()
    }
}
