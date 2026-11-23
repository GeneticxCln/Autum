// Autum Settings Window — model-agnostic live configuration UI.
// Pick any locally installed Ollama model for Brain/Chat, Vision, and Voice,
// adjust listening/privacy/desktop controls, and hot-reload Autum live.
import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls.Basic

Window {
    id: root
    title: "Autum Settings"
    width: 680
    height: 760
    minimumWidth: 540
    minimumHeight: 560
    visible: false
    transientParent: null
    flags: Qt.Window
    color: "#0b0f17"

    // Palette matching Autum's silver-ice holographic presence
    readonly property color bgCanvas: "#0b0f17"
    readonly property color bgCard: "#121926"
    readonly property color bgInput: "#0d131e"
    readonly property color bgHover: "#1b2638"
    readonly property color borderSubtle: "#223047"
    readonly property color borderFocus: "#7cc7e8"
    readonly property color textPrimary: "#e6eef8"
    readonly property color textSecondary: "#8ea3be"
    readonly property color accentIce: "#7cc7e8"
    readonly property color accentGlow: "#b4f0dc"
    readonly property color dangerSoft: "#e89aa6"

    property var installedModels: []
    property var allowedDirsList: []
    property var extraIntegrations: []
    property var diagnosticChecks: []
    property var diagnosticProblems: []
    property string diagnosticOverall: ""
    property string selectedMode: "push_to_talk"
    property string lastSyncedPresenceMode: ""
    property bool pendingScrollToDiagnostics: false
    property real loadedApprovalTimeout: 20.0
    property int loadedApprovalSpinValue: 20
    property string statusText: ""
    property bool statusError: false
    property string dataDirValue: ""

    function openSettings() {
        pendingScrollToDiagnostics = false
        loadFromDisk()
        visible = true
        raise()
        requestActivate()
    }

    function openDiagnostics() {
        pendingScrollToDiagnostics = true
        loadFromDisk()
        visible = true
        raise()
        requestActivate()
    }

    function runSelfDiagnosis() {
        if (typeof settingsBridge === "undefined" || !settingsBridge)
            return
        if (typeof settingsBridge.runDiagnosticsAsync === "function") {
            settingsBridge.runDiagnosticsAsync()
            return
        }
        const rep = settingsBridge.runDiagnostics()
        if (rep) {
            diagnosticOverall = rep.overall_status || "ok"
            diagnosticChecks = rep.checks || []
            diagnosticProblems = rep.recent_problems || []
            if (pendingScrollToDiagnostics) {
                pendingScrollToDiagnostics = false
                Qt.callLater(() => {
                    flick.contentY = Math.max(0, flick.contentHeight - flick.height)
                })
            }
        }
    }

    Connections {
        target: typeof presence !== "undefined" ? presence : null
        function onChanged() {
            if (!presence || !presence.mode)
                return
            if (presence.mode !== root.lastSyncedPresenceMode) {
                root.lastSyncedPresenceMode = presence.mode
                if (presence.mode === "push_to_talk" || presence.mode === "wake_word" || presence.mode === "hands_free")
                    root.selectedMode = presence.mode
            }
        }
    }

    Connections {
        target: typeof settingsBridge !== "undefined" ? settingsBridge : null
        function onDiagnosticsReady(rep) {
            if (rep) {
                root.diagnosticOverall = rep.overall_status || "ok"
                root.diagnosticChecks = rep.checks || []
                root.diagnosticProblems = rep.recent_problems || []
            }
            if (root.pendingScrollToDiagnostics) {
                root.pendingScrollToDiagnostics = false
                Qt.callLater(() => {
                    flick.contentY = Math.max(0, flick.contentHeight - flick.height)
                })
            }
        }
        function onModelsReady(models) {
            const prevChat = chatModelBox.editText
            const prevVision = visionModelBox.editText
            const prevVoice = voiceModelBox.editText
            root.installedModels = models || []
            chatModelBox.editText = prevChat
            visionModelBox.editText = prevVision
            voiceModelBox.editText = prevVoice
        }
    }

    function clearProblemLog() {
        if (typeof settingsBridge === "undefined" || !settingsBridge)
            return
        settingsBridge.clearProblems()
        runSelfDiagnosis()
        statusError = false
        statusText = "Cleared diagnostics problem log"
    }

    function applyDict(cfg) {
        chatModelBox.editText = cfg.chat_model || ""
        visionModelBox.editText = cfg.vision_model || ""
        voiceModelBox.editText = cfg.model || ""
        dynamicSwapSwitch.checked = Boolean(cfg.dynamic_swap)
        chatThinkSwitch.checked = Boolean(cfg.chat_think)
        ollamaUrlField.text = cfg.ollama_url || "http://127.0.0.1:11434"

        selectedMode = (cfg.mode === "hands_free" || cfg.mode === "wake_word") ? cfg.mode : "push_to_talk"
        if (typeof presence !== "undefined" && presence && presence.mode)
            lastSyncedPresenceMode = presence.mode
        aecSwitch.checked = Boolean(cfg.aec)
        voicePresetField.text = cfg.voice || "female-american-english"
        silenceSpin.value = Math.max(100, Math.min(5000, Number(cfg.silence_ms || 700)))
        bargeInSpin.value = Math.max(64, Math.min(2000, Number(cfg.barge_in_ms || 256)))

        allowedDirsList = (cfg.allowed_read_dirs || []).slice()
        const integrations = cfg.enabled_integrations || []
        desktopIntegrationSwitch.checked = integrations.indexOf("linux_desktop") >= 0
        extraIntegrations = integrations.filter(i => i !== "linux_desktop")
        reversibleApprovalSwitch.checked = Boolean(cfg.require_reversible_approval)
        const rawTimeout = Number(cfg.approval_timeout_s || 20)
        loadedApprovalTimeout = Math.max(2.0, Math.min(120.0, isNaN(rawTimeout) ? 20.0 : rawTimeout))
        loadedApprovalSpinValue = Math.max(2, Math.min(120, Math.round(loadedApprovalTimeout)))
        approvalTimeoutSpin.value = loadedApprovalSpinValue
        excludedWindowsField.text = (cfg.excluded_windows || []).join(", ")

        maxHistorySpin.value = Math.max(2, Math.min(200, Number(cfg.max_history || 20)))
        maxContextSpin.value = Math.max(500, Math.min(200000, Number(cfg.max_context_chars || 10000)))
        const lvl = String(cfg.log_level || "INFO").toUpperCase()
        const idx = logLevelBox.model.indexOf(lvl)
        logLevelBox.currentIndex = idx >= 0 ? idx : 1
        dataDirValue = cfg.data_dir || ""
    }

    function loadFromDisk() {
        statusText = ""
        statusError = false
        if (typeof settingsBridge === "undefined" || !settingsBridge)
            return
        const cfg = settingsBridge.loadSettings()
        applyDict(cfg)
        if (cfg && cfg._load_error) {
            statusError = true
            statusText = cfg._load_error
        }
        refreshModels()
        runSelfDiagnosis()
    }

    function refreshModels() {
        if (typeof settingsBridge === "undefined" || !settingsBridge)
            return
        if (typeof settingsBridge.listOllamaModelsAsync === "function") {
            settingsBridge.listOllamaModelsAsync(ollamaUrlField.text)
            return
        }
        const prevChat = chatModelBox.editText
        const prevVision = visionModelBox.editText
        const prevVoice = voiceModelBox.editText
        installedModels = settingsBridge.listOllamaModels(ollamaUrlField.text)
        chatModelBox.editText = prevChat
        visionModelBox.editText = prevVision
        voiceModelBox.editText = prevVoice
    }

    function addAllowedDir() {
        const cleaned = newDirField.text.trim()
        if (!cleaned)
            return
        if (allowedDirsList.indexOf(cleaned) < 0) {
            const copy = allowedDirsList.slice()
            copy.push(cleaned)
            allowedDirsList = copy
        }
        newDirField.text = ""
    }

    function removeAllowedDir(idx) {
        const copy = allowedDirsList.slice()
        copy.splice(idx, 1)
        allowedDirsList = copy
    }

    function saveSettings() {
        flick.forceActiveFocus()
        if (typeof settingsBridge === "undefined" || !settingsBridge)
            return
        const excluded = excludedWindowsField.text
            .split(",")
            .map(s => s.trim())
            .filter(s => s.length > 0)
        const integrations = desktopIntegrationSwitch.checked
            ? ["linux_desktop"].concat(extraIntegrations)
            : extraIntegrations.slice()
        const timeoutVal = (approvalTimeoutSpin.value === loadedApprovalSpinValue)
            ? loadedApprovalTimeout
            : approvalTimeoutSpin.value
        const payload = {
            "mode": selectedMode,
            "aec": aecSwitch.checked,
            "silence_ms": silenceSpin.value,
            "barge_in_ms": bargeInSpin.value,
            "max_history": maxHistorySpin.value,
            "max_context_chars": maxContextSpin.value,
            "ollama_url": ollamaUrlField.text.trim(),
            "model": voiceModelBox.editText.trim(),
            "chat_model": chatModelBox.editText.trim(),
            "vision_model": visionModelBox.editText.trim(),
            "dynamic_swap": dynamicSwapSwitch.checked,
            "chat_think": chatThinkSwitch.checked,
            "voice": voicePresetField.text.trim(),
            "log_level": logLevelBox.currentText,
            "data_dir": dataDirValue,
            "allowed_read_dirs": allowedDirsList,
            "enabled_integrations": integrations,
            "excluded_windows": excluded,
            "require_reversible_approval": reversibleApprovalSwitch.checked,
            "approval_timeout_s": timeoutVal
        }
        const res = settingsBridge.saveAndApply(payload)
        if (res && res.ok) {
            statusError = false
            const needsRestart = res.restart_required || []
            if (res.applied_live === false) {
                statusText = "Saved to disk (runtime offline — takes effect on next start)"
            } else {
                statusText = needsRestart.length > 0
                    ? ("Saved & applied (restart Autum for " + needsRestart.join(", ") + ")")
                    : "Saved & applied live"
            }
            refreshModels()
            runSelfDiagnosis()
        } else {
            statusError = true
            statusText = (res && res.error) ? res.error : "Failed to save settings"
        }
    }

    component SectionCard: Rectangle {
        default property alias body: innerCol.data
        property string heading: ""
        property string subtitle: ""
        Layout.fillWidth: true
        implicitHeight: outerCol.implicitHeight + 32
        radius: 14
        color: root.bgCard
        border.width: 1
        border.color: root.borderSubtle

        ColumnLayout {
            id: outerCol
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 16
            spacing: 12

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                visible: heading !== ""
                Text {
                    text: heading
                    color: root.textPrimary
                    font.pixelSize: 15
                    font.weight: Font.DemiBold
                }
                Text {
                    visible: subtitle !== ""
                    text: subtitle
                    color: root.textSecondary
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }

            ColumnLayout {
                id: innerCol
                Layout.fillWidth: true
                spacing: 12
            }
        }
    }

    component StyledTextField: TextField {
        color: root.textPrimary
        placeholderTextColor: "#5c708a"
        font.pixelSize: 13
        leftPadding: 11
        rightPadding: 11
        topPadding: 8
        bottomPadding: 8
        selectByMouse: true
        background: Rectangle {
            radius: 8
            color: root.bgInput
            border.width: 1
            border.color: parent.activeFocus ? root.borderFocus : root.borderSubtle
        }
    }

    component StyledSwitch: Switch {
        id: sw
        implicitWidth: 44
        implicitHeight: 24
        indicator: Rectangle {
            implicitWidth: 44
            implicitHeight: 24
            radius: 12
            color: sw.checked ? "#2b6e88" : "#1c2536"
            border.width: 1
            border.color: sw.checked ? root.accentIce : root.borderSubtle
            Behavior on color { ColorAnimation { duration: 150 } }

            Rectangle {
                x: sw.checked ? parent.width - width - 3 : 3
                anchors.verticalCenter: parent.verticalCenter
                width: 18
                height: 18
                radius: 9
                color: sw.checked ? "#eaf6ff" : "#8ea3be"
                Behavior on x { NumberAnimation { duration: 150; easing.type: Easing.OutCubic } }
            }
        }
    }

    component StyledSpinBox: SpinBox {
        id: sb
        editable: true
        font.pixelSize: 13
        contentItem: TextInput {
            text: sb.textFromValue(sb.value, sb.locale)
            font: sb.font
            color: root.textPrimary
            horizontalAlignment: Qt.AlignHCenter
            verticalAlignment: Qt.AlignVCenter
            readOnly: !sb.editable
            validator: sb.validator
            selectByMouse: true
        }
        up.indicator: Rectangle {
            x: sb.width - width
            height: sb.height
            implicitWidth: 28
            radius: 7
            color: sb.up.pressed ? root.bgHover : "transparent"
            Text {
                anchors.centerIn: parent
                text: "+"
                color: root.textPrimary
                font.pixelSize: 14
            }
        }
        down.indicator: Rectangle {
            x: 0
            height: sb.height
            implicitWidth: 28
            radius: 7
            color: sb.down.pressed ? root.bgHover : "transparent"
            Text {
                anchors.centerIn: parent
                text: "−"
                color: root.textPrimary
                font.pixelSize: 14
            }
        }
        background: Rectangle {
            implicitWidth: 128
            implicitHeight: 34
            radius: 8
            color: root.bgInput
            border.width: 1
            border.color: sb.activeFocus ? root.borderFocus : root.borderSubtle
        }
    }

    component ModelPicker: ColumnLayout {
        id: picker
        property string label: ""
        property string hint: ""
        property alias editText: combo.editText
        Layout.fillWidth: true
        spacing: 4

        Text {
            text: picker.label
            color: root.textPrimary
            font.pixelSize: 13
            font.weight: Font.Medium
        }

        ComboBox {
            id: combo
            Layout.fillWidth: true
            editable: true
            model: root.installedModels
            textRole: "name"
            font.pixelSize: 13
            onActivated: (index) => {
                if (index >= 0 && index < root.installedModels.length)
                    combo.editText = root.installedModels[index].name
            }

            contentItem: TextInput {
                leftPadding: 11
                rightPadding: 30
                text: combo.editText
                font: combo.font
                color: root.textPrimary
                verticalAlignment: Text.AlignVCenter
                selectByMouse: true
                onTextEdited: combo.editText = text
            }

            indicator: Text {
                x: combo.width - width - 11
                anchors.verticalCenter: parent.verticalCenter
                text: "▾"
                color: root.textSecondary
                font.pixelSize: 13
            }

            background: Rectangle {
                implicitHeight: 36
                radius: 8
                color: root.bgInput
                border.width: 1
                border.color: combo.activeFocus || combo.popup.visible ? root.borderFocus : root.borderSubtle
            }

            delegate: ItemDelegate {
                required property var modelData
                required property int index
                width: combo.width
                height: 36
                highlighted: combo.highlightedIndex === index
                background: Rectangle {
                    color: parent.highlighted ? root.bgHover : "#101724"
                }
                contentItem: RowLayout {
                    spacing: 8
                    Text {
                        text: modelData.name || ""
                        color: root.textPrimary
                        font.pixelSize: 13
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                    }
                    Text {
                        text: (modelData.size_gb ? (modelData.size_gb + " GB") : "")
                              + (modelData.status ? (" • " + modelData.status) : "")
                        color: modelData.loaded ? root.accentGlow : root.textSecondary
                        font.pixelSize: 11
                    }
                }
            }

            popup: Popup {
                y: combo.height + 4
                width: combo.width
                implicitHeight: Math.min(240, contentItem.implicitHeight + 8)
                padding: 4
                background: Rectangle {
                    color: "#101724"
                    radius: 10
                    border.width: 1
                    border.color: root.borderSubtle
                }
                contentItem: ListView {
                    clip: true
                    implicitHeight: contentHeight
                    model: combo.delegateModel
                    currentIndex: combo.highlightedIndex
                }
            }
        }

        Text {
            visible: picker.hint !== ""
            text: picker.hint
            color: root.textSecondary
            font.pixelSize: 11
            wrapMode: Text.WordWrap
            Layout.fillWidth: true
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 70
            color: "#0f1521"
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: 1
                color: root.borderSubtle
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 12

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 2
                    RowLayout {
                        spacing: 8
                        Text {
                            text: "Autum Settings"
                            color: root.textPrimary
                            font.pixelSize: 19
                            font.weight: Font.DemiBold
                        }
                        Rectangle {
                            radius: 9
                            implicitHeight: 20
                            implicitWidth: verLabel.implicitWidth + 14
                            color: "#172233"
                            border.width: 1
                            border.color: root.borderSubtle
                            Text {
                                id: verLabel
                                anchors.centerIn: parent
                                text: "v" + ((typeof settingsBridge !== "undefined" && settingsBridge && typeof settingsBridge.appVersion === "function") ? settingsBridge.appVersion() : "1.0.0")
                                color: root.accentIce
                                font.pixelSize: 11
                                font.weight: Font.Medium
                            }
                        }
                    }
                    Text {
                        text: "Model-agnostic local assistant — pick any Ollama models and apply live"
                        color: root.textSecondary
                        font.pixelSize: 12
                    }
                }

                Rectangle {
                    radius: 12
                    color: presence.connected ? "#132931" : "#261820"
                    border.width: 1
                    border.color: presence.connected ? "#2c6b78" : "#5e3340"
                    implicitHeight: 26
                    implicitWidth: liveLabel.implicitWidth + 20
                    Text {
                        id: liveLabel
                        anchors.centerIn: parent
                        text: presence.connected
                              ? ("● Live • " + (presence.mode === "hands_free"
                                                ? "Hands-Free"
                                                : (presence.mode === "wake_word" ? "Wake Word (\"Autum\")" : "Push-to-Talk")))
                              : "○ Runtime Offline"
                        color: presence.connected ? root.accentGlow : root.dangerSoft
                        font.pixelSize: 11
                        font.weight: Font.Medium
                    }
                }
            }
        }

        // Scrollable Settings Body
        Flickable {
            id: flick
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: width
            contentHeight: mainCol.implicitHeight + 36
            boundsBehavior: Flickable.StopAtBounds

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
                contentItem: Rectangle {
                    implicitWidth: 6
                    radius: 3
                    color: "#2c3d57"
                }
            }

            ColumnLayout {
                id: mainCol
                width: flick.width - 36
                x: 18
                y: 18
                spacing: 16

                // 1. AI Models (3 Pickers)
                SectionCard {
                    heading: "AI Models (Ollama)"
                    subtitle: "Autum does not bundle models. Pick any model pulled into your local Ollama instance for each role, or type a custom tag."

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Text {
                            Layout.fillWidth: true
                            color: root.installedModels.length > 0 ? root.accentGlow : root.dangerSoft
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                            text: {
                                if (root.installedModels.length === 0)
                                    return "No Ollama models detected at " + ollamaUrlField.text
                                const loaded = root.installedModels
                                    .filter(m => m.loaded)
                                    .map(m => m.name + " (" + m.status + ")")
                                return root.installedModels.length + " local models found"
                                       + (loaded.length > 0 ? "  •  In VRAM: " + loaded.join(", ") : "")
                            }
                        }
                        Button {
                            text: "Refresh Models"
                            onClicked: root.refreshModels()
                            contentItem: Text {
                                text: parent.text
                                color: root.accentIce
                                font.pixelSize: 12
                                font.weight: Font.Medium
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle {
                                implicitHeight: 30
                                implicitWidth: 116
                                radius: 8
                                color: parent.hovered ? root.bgHover : root.bgInput
                                border.width: 1
                                border.color: root.borderSubtle
                            }
                        }
                    }

                    ModelPicker {
                        id: chatModelBox
                        label: "Brain / Chat Model"
                        hint: "Main reasoning, conversation, memory, and tool calling (tested with gemma4:12b, qwen3:8b, lfm3-omni:2b)."
                    }

                    ModelPicker {
                        id: visionModelBox
                        label: "Vision Model"
                        hint: "Used when looking at your screen, windows, or images (tested with qwen3-vl:8b, gemma4:12b)."
                    }

                    ModelPicker {
                        id: voiceModelBox
                        label: "Voice / Speech Model"
                        hint: "Handles speech-to-text and natural voice synthesis (tested with lfm3-omni:2b)."
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Dynamic VRAM Swap"; color: root.textPrimary; font.pixelSize: 13 }
                            Text {
                                text: "Unload Brain when Vision runs (and vice versa) so 16 GB GPUs keep models 100% in VRAM."
                                color: root.textSecondary
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        StyledSwitch { id: dynamicSwapSwitch }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Model Thinking Mode"; color: root.textPrimary; font.pixelSize: 13 }
                            Text {
                                text: "Enable internal chain-of-thought reasoning on supporting chat models (slower, deeper answers)."
                                color: root.textSecondary
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        StyledSwitch { id: chatThinkSwitch }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Text { text: "Ollama Server URL (localhost only)"; color: root.textPrimary; font.pixelSize: 13 }
                        StyledTextField {
                            id: ollamaUrlField
                            Layout.fillWidth: true
                            placeholderText: "http://127.0.0.1:11434"
                        }
                    }
                }

                // 2. Listening & Voice
                SectionCard {
                    heading: "Listening & Voice"
                    subtitle: "Choose Push-to-Talk (hold orb with mouse), Wake Word (only responds when you say 'Autum'), or full Hands-Free."

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        Text {
                            text: "Listening Mode"
                            color: root.textPrimary
                            font.pixelSize: 13
                            Layout.fillWidth: true
                        }
                        Row {
                            spacing: 6
                            Repeater {
                                model: [
                                    { "id": "push_to_talk", "label": "Push-to-Talk" },
                                    { "id": "wake_word", "label": "Wake (\"Autum\")" },
                                    { "id": "hands_free", "label": "Hands-Free" }
                                ]
                                delegate: Button {
                                    required property var modelData
                                    readonly property bool active: root.selectedMode === modelData.id
                                    text: modelData.label
                                    onClicked: root.selectedMode = modelData.id
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.active ? "#0b0f17" : root.textPrimary
                                        font.pixelSize: 12
                                        font.weight: parent.active ? Font.DemiBold : Font.Normal
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        implicitHeight: 32
                                        implicitWidth: 110
                                        radius: 8
                                        color: parent.active ? root.accentIce : (parent.hovered ? root.bgHover : root.bgInput)
                                        border.width: 1
                                        border.color: parent.active ? root.accentIce : root.borderSubtle
                                    }
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "PipeWire Echo Cancellation (AEC)"; color: root.textPrimary; font.pixelSize: 13 }
                            Text {
                                text: "Filter speaker playback out of your microphone during Hands-Free sessions."
                                color: root.textSecondary
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        StyledSwitch { id: aecSwitch }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Text { text: "Voice Preset"; color: root.textPrimary; font.pixelSize: 13 }
                        StyledTextField {
                            id: voicePresetField
                            Layout.fillWidth: true
                            placeholderText: "female-american-english"
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: "End-of-turn Silence (ms)"; color: root.textPrimary; font.pixelSize: 12 }
                            StyledSpinBox { id: silenceSpin; from: 100; to: 5000; stepSize: 50; Layout.fillWidth: true }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: "Barge-in Onset (ms)"; color: root.textPrimary; font.pixelSize: 12 }
                            StyledSpinBox { id: bargeInSpin; from: 64; to: 2000; stepSize: 32; Layout.fillWidth: true }
                        }
                    }
                }

                // 3. Privacy, Folders & Desktop
                SectionCard {
                    heading: "Privacy, Folders & Desktop"
                    subtitle: "Control which folders Autum can read, desktop Eyes & Hands permissions, and approval prompts."

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Text { text: "Allowed Read Folders"; color: root.textPrimary; font.pixelSize: 13 }
                        Text {
                            text: root.allowedDirsList.length === 0
                                  ? "None configured (fail-closed: Autum cannot read local files until you add a folder)."
                                  : "Autum may read documents, images, and folders inside these paths:"
                            color: root.textSecondary
                            font.pixelSize: 11
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }

                        Repeater {
                            model: root.allowedDirsList
                            delegate: Rectangle {
                                required property string modelData
                                required property int index
                                Layout.fillWidth: true
                                implicitHeight: 32
                                radius: 8
                                color: root.bgInput
                                border.width: 1
                                border.color: root.borderSubtle

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    anchors.rightMargin: 6
                                    Text {
                                        text: modelData
                                        color: root.textPrimary
                                        font.pixelSize: 12
                                        elide: Text.ElideMiddle
                                        Layout.fillWidth: true
                                    }
                                    Button {
                                        text: "×"
                                        onClicked: root.removeAllowedDir(index)
                                        contentItem: Text {
                                            text: "×"
                                            color: root.dangerSoft
                                            font.pixelSize: 15
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter
                                        }
                                        background: Rectangle {
                                            implicitWidth: 24
                                            implicitHeight: 24
                                            radius: 6
                                            color: parent.hovered ? "#2a1b24" : "transparent"
                                        }
                                    }
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            StyledTextField {
                                id: newDirField
                                Layout.fillWidth: true
                                placeholderText: "/home/you/Documents"
                                onAccepted: root.addAllowedDir()
                            }
                            Button {
                                text: "Add Folder"
                                onClicked: root.addAllowedDir()
                                contentItem: Text {
                                    text: parent.text
                                    color: root.textPrimary
                                    font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle {
                                    implicitHeight: 34
                                    implicitWidth: 94
                                    radius: 8
                                    color: parent.hovered ? root.bgHover : root.bgInput
                                    border.width: 1
                                    border.color: root.borderSubtle
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Linux Desktop Integration (Eyes & Hands)"; color: root.textPrimary; font.pixelSize: 13 }
                            Text {
                                text: "Allow screen/window inspection and window/app management (restart required to toggle plugin registration)."
                                color: root.textSecondary
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        StyledSwitch { id: desktopIntegrationSwitch }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            Text { text: "Require Approval for Reversible Actions"; color: root.textPrimary; font.pixelSize: 13 }
                            Text {
                                text: "Destructive actions always require approval. Turn this on to also confirm opening apps or moving windows."
                                color: root.textSecondary
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                        StyledSwitch { id: reversibleApprovalSwitch }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        Text {
                            text: "Approval Timeout (seconds)"
                            color: root.textPrimary
                            font.pixelSize: 13
                            Layout.fillWidth: true
                        }
                        StyledSpinBox { id: approvalTimeoutSpin; from: 2; to: 120; stepSize: 5 }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Text { text: "Excluded Sensitive Windows (comma-separated)"; color: root.textPrimary; font.pixelSize: 13 }
                        StyledTextField {
                            id: excludedWindowsField
                            Layout.fillWidth: true
                            placeholderText: "keepassxc, 1password, bitwarden, private browsing"
                        }
                    }
                }

                // 4. Context & Logging
                SectionCard {
                    heading: "Context & Logging"

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: "Max History Turns"; color: root.textPrimary; font.pixelSize: 12 }
                            StyledSpinBox { id: maxHistorySpin; from: 2; to: 200; Layout.fillWidth: true }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: "Max Context Chars"; color: root.textPrimary; font.pixelSize: 12 }
                            StyledSpinBox { id: maxContextSpin; from: 500; to: 200000; stepSize: 1000; Layout.fillWidth: true }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            Text { text: "Log Level"; color: root.textPrimary; font.pixelSize: 12 }
                            ComboBox {
                                id: logLevelBox
                                Layout.fillWidth: true
                                model: ["DEBUG", "INFO", "WARNING", "ERROR"]
                                font.pixelSize: 13
                                contentItem: Text {
                                    leftPadding: 11
                                    text: logLevelBox.displayText
                                    font: logLevelBox.font
                                    color: root.textPrimary
                                    verticalAlignment: Text.AlignVCenter
                                }
                                indicator: Text {
                                    x: logLevelBox.width - width - 10
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "▾"
                                    color: root.textSecondary
                                }
                                background: Rectangle {
                                    implicitHeight: 34
                                    radius: 8
                                    color: root.bgInput
                                    border.width: 1
                                    border.color: logLevelBox.activeFocus ? root.borderFocus : root.borderSubtle
                                }
                                delegate: ItemDelegate {
                                    required property string modelData
                                    required property int index
                                    width: logLevelBox.width
                                    highlighted: logLevelBox.highlightedIndex === index
                                    background: Rectangle {
                                        color: parent.highlighted ? root.bgHover : "#101724"
                                    }
                                    contentItem: Text {
                                        text: modelData
                                        color: root.textPrimary
                                        font.pixelSize: 13
                                    }
                                }
                                popup: Popup {
                                    y: logLevelBox.height + 4
                                    width: logLevelBox.width
                                    implicitHeight: contentItem.implicitHeight + 8
                                    padding: 4
                                    background: Rectangle {
                                        color: "#101724"
                                        radius: 8
                                        border.width: 1
                                        border.color: root.borderSubtle
                                    }
                                    contentItem: ListView {
                                        clip: true
                                        implicitHeight: contentHeight
                                        model: logLevelBox.delegateModel
                                    }
                                }
                            }
                        }
                    }
                }

                // 5. Self-Diagnosis & Problem Log
                SectionCard {
                    id: diagCard
                    heading: "Self-Diagnosis & Problem Log"
                    subtitle: "Live health checks across Ollama/VRAM, PipeWire microphone, desktop Eyes & Hands, permissions, and problems logged by Autum (~/.local/share/autum/diagnostics.jsonl)."

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Text {
                            Layout.fillWidth: true
                            font.pixelSize: 12
                            font.weight: Font.Medium
                            color: root.diagnosticOverall === "error"
                                   ? root.dangerSoft
                                   : (root.diagnosticOverall === "warning" ? "#ebd39b" : root.accentGlow)
                            text: {
                                if (root.diagnosticChecks.length === 0)
                                    return "Click Run Self-Diagnosis to inspect all systems."
                                const label = root.diagnosticOverall === "error"
                                    ? "Action Required"
                                    : (root.diagnosticOverall === "warning" ? "Healthy with Warnings" : "All Systems Healthy")
                                return "Status: " + label + "  •  Logged problems: " + root.diagnosticProblems.length
                            }
                        }

                        Button {
                            text: "Run Self-Diagnosis"
                            onClicked: root.runSelfDiagnosis()
                            contentItem: Text {
                                text: parent.text
                                color: root.accentIce
                                font.pixelSize: 12
                                font.weight: Font.Medium
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle {
                                implicitHeight: 30
                                implicitWidth: 138
                                radius: 8
                                color: parent.hovered ? root.bgHover : root.bgInput
                                border.width: 1
                                border.color: root.borderSubtle
                            }
                        }

                        Button {
                            visible: root.diagnosticProblems.length > 0
                            text: "Clear Log"
                            onClicked: root.clearProblemLog()
                            contentItem: Text {
                                text: parent.text
                                color: root.dangerSoft
                                font.pixelSize: 12
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                            background: Rectangle {
                                implicitHeight: 30
                                implicitWidth: 84
                                radius: 8
                                color: parent.hovered ? "#2a1b24" : root.bgInput
                                border.width: 1
                                border.color: root.borderSubtle
                            }
                        }
                    }

                    // Subsystem health check cards
                    Repeater {
                        model: root.diagnosticChecks
                        delegate: Rectangle {
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: checkCol.implicitHeight + 18
                            radius: 8
                            color: root.bgInput
                            border.width: 1
                            border.color: modelData.status === "error"
                                          ? "#6b2d39"
                                          : (modelData.status === "warning" ? "#5c4d26" : root.borderSubtle)

                            ColumnLayout {
                                id: checkCol
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 9
                                spacing: 3

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Rectangle {
                                        radius: 6
                                        implicitWidth: 46
                                        implicitHeight: 18
                                        color: modelData.status === "error"
                                               ? "#3d1a22"
                                               : (modelData.status === "warning" ? "#382f16" : "#13332b")
                                        Text {
                                            anchors.centerIn: parent
                                            text: (modelData.status || "ok").toUpperCase()
                                            font.pixelSize: 10
                                            font.weight: Font.DemiBold
                                            color: modelData.status === "error"
                                                   ? root.dangerSoft
                                                   : (modelData.status === "warning" ? "#ebd39b" : root.accentGlow)
                                        }
                                    }
                                    Text {
                                        text: modelData.title || ""
                                        color: root.textPrimary
                                        font.pixelSize: 12
                                        font.weight: Font.DemiBold
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.summary || ""
                                        color: root.textSecondary
                                        font.pixelSize: 12
                                        elide: Text.ElideRight
                                    }
                                }

                                Text {
                                    visible: Boolean(modelData.details)
                                    text: modelData.details || ""
                                    color: root.textSecondary
                                    font.pixelSize: 11
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }

                                Text {
                                    visible: Boolean(modelData.suggestion)
                                    text: "Fix: " + (modelData.suggestion || "")
                                    color: root.accentIce
                                    font.pixelSize: 11
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }

                    // Logged problems list
                    Text {
                        text: root.diagnosticProblems.length > 0
                              ? "Recent Logged Problems (from runtime & Autum agent):"
                              : "No runtime or agent problems logged in diagnostics.jsonl."
                        color: root.textPrimary
                        font.pixelSize: 12
                        font.weight: Font.Medium
                    }

                    Repeater {
                        model: root.diagnosticProblems
                        delegate: Rectangle {
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: probCol.implicitHeight + 16
                            radius: 8
                            color: "#0e1522"
                            border.width: 1
                            border.color: modelData.severity === "error" ? "#6b2d39" : "#2b3b54"

                            ColumnLayout {
                                id: probCol
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 8
                                spacing: 2

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6
                                    Text {
                                        text: "[" + (modelData.severity || "warning").toUpperCase() + "] " + (modelData.component || "general")
                                        color: modelData.severity === "error" ? root.dangerSoft : "#ebd39b"
                                        font.pixelSize: 11
                                        font.weight: Font.DemiBold
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        horizontalAlignment: Text.AlignRight
                                        text: (modelData.timestamp || "") + (modelData.source ? (" • " + modelData.source) : "")
                                        color: root.textSecondary
                                        font.pixelSize: 10
                                    }
                                }

                                Text {
                                    text: modelData.summary || ""
                                    color: root.textPrimary
                                    font.pixelSize: 12
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }

                                Text {
                                    visible: Boolean(modelData.details)
                                    text: modelData.details || ""
                                    color: root.textSecondary
                                    font.pixelSize: 11
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }

                                Text {
                                    visible: Boolean(modelData.suggestion)
                                    text: "Suggested fix: " + (modelData.suggestion || "")
                                    color: root.accentIce
                                    font.pixelSize: 11
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sticky Footer
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 62
            color: "#0f1521"
            Rectangle {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                height: 1
                color: root.borderSubtle
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 10

                Button {
                    text: "Reset Defaults"
                    onClicked: {
                        if (typeof settingsBridge !== "undefined" && settingsBridge) {
                            root.applyDict(settingsBridge.defaultSettings())
                            root.statusError = false
                            root.statusText = "Defaults loaded (click Save & Apply)"
                        }
                    }
                    contentItem: Text {
                        text: parent.text
                        color: root.textSecondary
                        font.pixelSize: 12
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    background: Rectangle {
                        implicitHeight: 34
                        implicitWidth: 112
                        radius: 8
                        color: parent.hovered ? root.bgHover : "transparent"
                        border.width: 1
                        border.color: root.borderSubtle
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: root.statusText
                    color: root.statusError ? root.dangerSoft : root.accentGlow
                    font.pixelSize: 12
                    elide: Text.ElideRight
                    horizontalAlignment: Text.AlignHCenter
                }

                Button {
                    text: "Close"
                    onClicked: root.visible = false
                    contentItem: Text {
                        text: parent.text
                        color: root.textPrimary
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    background: Rectangle {
                        implicitHeight: 36
                        implicitWidth: 84
                        radius: 8
                        color: parent.hovered ? root.bgHover : root.bgInput
                        border.width: 1
                        border.color: root.borderSubtle
                    }
                }

                Button {
                    text: "Save & Apply"
                    onClicked: root.saveSettings()
                    contentItem: Text {
                        text: parent.text
                        color: "#0b0f17"
                        font.pixelSize: 13
                        font.weight: Font.DemiBold
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    background: Rectangle {
                        implicitHeight: 36
                        implicitWidth: 124
                        radius: 8
                        color: parent.pressed ? "#5faec9" : (parent.hovered ? "#9ad8f2" : root.accentIce)
                    }
                }
            }
        }
    }
}
