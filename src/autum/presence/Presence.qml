// Autum's presence (WP4): a 3D stippled face suspended inside a frosted
// stardust orb, draggable anywhere on screen with a right-click menu for
// Hands-Free mode and Settings.
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
import QtQuick.Controls.Basic
import Autum 1.0

Window {
    id: win
    // Keep the physical pixel size consistent between unscaled X11/XWayland
    // and native Wayland HiDPI (e.g. 1.5x scale on 4K).
    readonly property real px: 1 / Math.max(1, Screen.devicePixelRatio)
    readonly property int size: Math.round(300 * px)   // orb diameter at rest, in physical pixels
    readonly property int pad: Math.round(size * 0.25) // room for voice burst & edge auras
    readonly property int margin: Math.round(24 * px)
    property int marginRight: margin - pad
    property int marginBottom: margin - pad
    width: size + 2 * pad
    height: width
    x: Screen.virtualX + Screen.desktopAvailableWidth - width - marginRight
    y: Screen.virtualY + Screen.desktopAvailableHeight - height - marginBottom
    visible: false  // shown by __main__.py
    title: "Autum"
    color: "transparent"
    flags: Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowDoesNotAcceptFocus

    function clampMarginRight(val) {
        const minM = -pad
        const availW = Screen.desktopAvailableWidth > size ? Screen.desktopAvailableWidth : (Screen.width > size ? Screen.width : 3840)
        const maxM = Math.max(minM, availW - size - pad)
        return Math.max(minM, Math.min(maxM, Math.round(val)))
    }

    function clampMarginBottom(val) {
        const minM = -pad
        const availH = Screen.desktopAvailableHeight > size ? Screen.desktopAvailableHeight : (Screen.height > size ? Screen.height : 2160)
        const maxM = Math.max(minM, availH - size - pad)
        return Math.max(minM, Math.min(maxM, Math.round(val)))
    }

    function resetOrbPosition() {
        marginRight = margin - pad
        marginBottom = margin - pad
        if (typeof settingsBridge !== "undefined" && settingsBridge)
            settingsBridge.resetPosition()
    }

    Component.onCompleted: {
        if (typeof settingsBridge !== "undefined" && settingsBridge) {
            const pos = settingsBridge.loadPosition()
            if (pos && pos.saved) {
                marginRight = clampMarginRight(pos.margin_right)
                marginBottom = clampMarginBottom(pos.margin_bottom)
            }
        }
    }

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

    // Time-driven phases updated at 30 Hz (also commits smooth drag moves per frame).
    property real spin: 0
    property real swirl: 0
    property real phase: 0
    property real lastTick: Date.now()
    Timer {
        interval: (win.st === "idle" && !talkArea.hoveringOrb && !talkArea.dragging && !win.observing && !win.micOpen) ? 50 : 33
        running: true
        repeat: true
        onTriggered: {
            const now = Date.now()
            const dt = Math.min((now - win.lastTick) / 1000, 0.1)
            win.lastTick = now
            win.spin += win.spinSpeed * dt
            win.swirl += win.swirlSpeed * dt
            win.phase += dt
            if (talkArea.dragging && talkArea.hasDragDelta && !talkArea.awaitingCommit) {
                const dx = Math.round(talkArea.curX - talkArea.pressX)
                const dy = Math.round(talkArea.curY - talkArea.pressY)
                talkArea.hasDragDelta = false
                if (dx !== 0 || dy !== 0) {
                    const oldMR = win.marginRight
                    const oldMB = win.marginBottom
                    const newMR = win.clampMarginRight(oldMR - dx)
                    const newMB = win.clampMarginBottom(oldMB - dy)
                    const appliedDx = oldMR - newMR
                    const appliedDy = oldMB - newMB
                    if (appliedDx !== 0 || appliedDy !== 0) {
                        talkArea.awaitingCommit = true
                        win.marginRight = newMR
                        win.marginBottom = newMB
                    }
                    // Keep anchor synced when clamped against a screen edge so reversing never jumps.
                    talkArea.pressX = Math.max(0, Math.min(win.width, talkArea.pressX + (dx - appliedDx)))
                    talkArea.pressY = Math.max(0, Math.min(win.height, talkArea.pressY + (dy - appliedDy)))
                    talkArea.curX = talkArea.pressX
                    talkArea.curY = talkArea.pressY
                }
            }
            faceCloud.redraw()
        }
    }

    Connections {
        target: win
        function onFrameSwapped() {
            if (talkArea.awaitingCommit) {
                talkArea.awaitingCommit = false
                talkArea.hasDragDelta = false
            }
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
        hovering: talkArea.hoveringOrb && !talkArea.dragging
    }

    Timer {
        id: dragGraceTalkTimer
        interval: 140
        repeat: false
        onTriggered: {
            if (talkArea.leftHeld && !talkArea.dragging && !talkArea.talking) {
                talkArea.talking = true
                talkArea.talkLocked = true
                presence.talkStart()
            }
        }
    }

    Timer {
        id: talkLockTimer
        interval: 220
        repeat: false
        onTriggered: {
            if (talkArea.leftHeld && talkArea.talking && !talkArea.dragging)
                talkArea.talkLocked = true
        }
    }

    Timer {
        id: tapTalkTimer
        interval: 22000
        repeat: false
        onTriggered: {
            if (talkArea.tapTalking) {
                talkArea.tapTalking = false
                presence.talkFinish()
            }
        }
    }

    onStChanged: {
        if (st !== "speech_detected" && st !== "listening" && talkArea.tapTalking && !talkArea.leftHeld) {
            tapTalkTimer.stop()
            talkArea.tapTalking = false
        }
    }

    // Left-drag moves the orb anywhere on screen; holding talks immediately at
    // 0 ms when idle (or after a brief 140 ms drag grace window while Autum is
    // speaking so dragging the orb out of the way never interrupts her reply).
    // Once held for 220 ms without dragging, talk hold locks so hand movement
    // while speaking never aborts the utterance; a quick single tap (< 240 ms)
    // keeps recording for 3 s (or until tapped again).
    MouseArea {
        id: talkArea
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        cursorShape: dragging ? Qt.ClosedHandCursor : (hoveringOrb ? Qt.OpenHandCursor : Qt.ArrowCursor)
        property real lookX: 0
        property real lookY: 0
        property bool hoveringOrb: false
        property bool leftHeld: false
        property bool dragging: false
        property bool talking: false
        property bool talkLocked: false
        property bool tapTalking: false
        property bool finishTapOnRelease: false
        property real pressTimeMs: 0
        property bool hasDragDelta: false
        property bool awaitingCommit: false
        property real pressX: 0
        property real pressY: 0
        property real curX: 0
        property real curY: 0

        onPositionChanged: (mouse) => {
            const dx = (mouse.x - width / 2) / win.radius
            const dy = (mouse.y - height / 2) / win.radius
            hoveringOrb = Math.sqrt(dx * dx + dy * dy) <= 1.15
            if (hoveringOrb && !dragging) {
                lookX = Math.max(-1, Math.min(1, dx))
                lookY = Math.max(-1, Math.min(1, dy))
            }
            if (leftHeld) {
                if (awaitingCommit)
                    return
                curX = mouse.x
                curY = mouse.y
                if (!dragging && !talkLocked) {
                    const moveDist = Math.hypot(mouse.x - pressX, mouse.y - pressY)
                    const dragThreshold = talking ? 18 : 8
                    if (moveDist >= dragThreshold) {
                        dragGraceTalkTimer.stop()
                        talkLockTimer.stop()
                        tapTalkTimer.stop()
                        tapTalking = false
                        finishTapOnRelease = false
                        dragging = true
                        if (talking) {
                            talking = false
                            presence.talkAbort()
                        }
                    }
                }
                if (dragging)
                    hasDragDelta = true
            }
        }
        onExited: hoveringOrb = false
        onPressed: (mouse) => {
            const dx = mouse.x - width / 2, dy = mouse.y - height / 2
            if (Math.sqrt(dx * dx + dy * dy) > win.radius * 1.05) {
                mouse.accepted = false
                return
            }
            if (mouse.button === Qt.RightButton) {
                contextMenu.popup(mouse.x, mouse.y)
                return
            }
            pressX = mouse.x
            pressY = mouse.y
            curX = mouse.x
            curY = mouse.y
            pressTimeMs = Date.now()
            hasDragDelta = false
            awaitingCommit = false
            leftHeld = true
            dragging = false
            talkLocked = false
            finishTapOnRelease = false
            if (tapTalking) {
                tapTalkTimer.stop()
                tapTalking = false
                talking = false
                finishTapOnRelease = true
                presence.talkFinish()
                return
            }
            // In push_to_talk mode (or when answering an approval prompt):
            // - At rest, start recording at 0 ms so the first syllable is never clipped.
            // - While speaking/busy, use a 140 ms drag grace window so dragging the orb
            //   out of the way never interrupts Autum's reply or opens the microphone.
            if (presence.mode === "push_to_talk" || win.st === "waiting_for_approval") {
                if (win.speaking || win.busy) {
                    talking = false
                    dragGraceTalkTimer.restart()
                } else {
                    talking = true
                    talkLockTimer.restart()
                    presence.talkStart()
                }
            } else {
                talking = false
            }
        }
        onReleased: (mouse) => {
            if (mouse.button !== Qt.LeftButton)
                return
            dragGraceTalkTimer.stop()
            talkLockTimer.stop()
            const heldMs = Date.now() - pressTimeMs
            if (dragging) {
                if (hasDragDelta && !awaitingCommit) {
                    const dx = Math.round(curX - pressX)
                    const dy = Math.round(curY - pressY)
                    if (dx !== 0 || dy !== 0) {
                        win.marginRight = win.clampMarginRight(win.marginRight - dx)
                        win.marginBottom = win.clampMarginBottom(win.marginBottom - dy)
                    }
                }
                if (typeof settingsBridge !== "undefined" && settingsBridge)
                    settingsBridge.savePosition(win.marginRight, win.marginBottom)
            } else if (finishTapOnRelease) {
                finishTapOnRelease = false
            } else if (talking) {
                if (heldMs < 240) {
                    tapTalking = true
                    tapTalkTimer.restart()
                    presence.talkAutoFinish()
                } else {
                    presence.talkFinish()
                }
            } else if (win.speaking || win.busy) {
                presence.stopSpeaking()
            }
            leftHeld = false
            dragging = false
            talking = false
            talkLocked = false
            hasDragDelta = false
            awaitingCommit = false
        }
        onCanceled: {
            dragGraceTalkTimer.stop()
            talkLockTimer.stop()
            tapTalkTimer.stop()
            tapTalking = false
            finishTapOnRelease = false
            if (talking)
                presence.talkAbort()
            leftHeld = false
            dragging = false
            talking = false
            talkLocked = false
            hasDragDelta = false
            awaitingCommit = false
        }
    }

    // Floating dark-glass context menu (Popup.Window escapes the circular orb mask)
    Menu {
        id: contextMenu
        popupType: Popup.Window
        width: 248
        padding: 6

        function switchMode(targetMode) {
            if (typeof settingsBridge !== "undefined" && settingsBridge)
                settingsBridge.setModeAndPersist(targetMode)
            else
                presence.setMode(targetMode)
        }

        background: Rectangle {
            color: "#101724"
            radius: 12
            border.width: 1
            border.color: "#26354d"
        }

        component StyledMenuItem: MenuItem {
            id: mi
            property string badge: ""
            property bool badgeActive: false
            implicitHeight: 36
            leftPadding: 12
            rightPadding: 12
            background: Rectangle {
                radius: 8
                color: mi.highlighted && mi.enabled ? "#1b283c" : "transparent"
            }
            contentItem: Row {
                spacing: 8
                Text {
                    text: mi.text
                    color: mi.enabled ? "#e6eef8" : "#5c708a"
                    font.pixelSize: 13
                    anchors.verticalCenter: parent.verticalCenter
                    width: mi.width - 24 - (mi.badge !== "" ? 50 : 0)
                    elide: Text.ElideRight
                }
                Rectangle {
                    visible: mi.badge !== ""
                    width: 42
                    height: 20
                    radius: 10
                    anchors.verticalCenter: parent.verticalCenter
                    color: mi.badgeActive ? "#1d4f60" : "#182132"
                    border.width: 1
                    border.color: mi.badgeActive ? "#7cc7e8" : "#2c3d57"
                    Text {
                        anchors.centerIn: parent
                        text: mi.badge
                        color: mi.badgeActive ? "#b4f0dc" : "#8ea3be"
                        font.pixelSize: 10
                        font.weight: Font.DemiBold
                    }
                }
            }
        }

        StyledMenuItem {
            text: "Push-to-Talk (Mouse)"
            badge: presence.mode === "push_to_talk" ? "ON" : ""
            badgeActive: presence.mode === "push_to_talk"
            onTriggered: contextMenu.switchMode("push_to_talk")
        }

        StyledMenuItem {
            text: "Wake Word (\"Autum\")"
            badge: presence.mode === "wake_word" ? "ON" : ""
            badgeActive: presence.mode === "wake_word"
            onTriggered: contextMenu.switchMode("wake_word")
        }

        StyledMenuItem {
            text: "Hands-Free"
            badge: presence.mode === "hands_free" ? "ON" : ""
            badgeActive: presence.mode === "hands_free"
            onTriggered: contextMenu.switchMode("hands_free")
        }

        MenuSeparator {
            padding: 4
            contentItem: Rectangle {
                implicitHeight: 1
                color: "#223047"
            }
        }

        StyledMenuItem {
            text: "Stop Speaking"
            enabled: win.speaking || win.busy
            onTriggered: presence.stopSpeaking()
        }

        MenuSeparator {
            padding: 4
            contentItem: Rectangle {
                implicitHeight: 1
                color: "#223047"
            }
        }

        StyledMenuItem {
            text: "Self-Diagnose..."
            onTriggered: settingsWindow.openDiagnostics()
        }

        StyledMenuItem {
            text: "Settings..."
            onTriggered: settingsWindow.openSettings()
        }

        StyledMenuItem {
            text: "Reset Position"
            onTriggered: win.resetOrbPosition()
        }

        MenuSeparator {
            padding: 4
            contentItem: Rectangle {
                implicitHeight: 1
                color: "#223047"
            }
        }

        StyledMenuItem {
            text: "Reload Bubble"
            onTriggered: presence.reloadBubble()
        }

        StyledMenuItem {
            text: "Close Bubble"
            onTriggered: presence.closeBubble()
        }
    }

    SettingsWindow {
        id: settingsWindow
    }
}
