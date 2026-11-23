#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')}"
OUT_DIR="${ROOT}/dist"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${OUT_DIR}"
APPDIR="${STAGE}/Autum.AppDir"
mkdir -p \
  "${APPDIR}/usr/bin" \
  "${APPDIR}/usr/share/applications" \
  "${APPDIR}/usr/share/icons/hicolor/256x256/apps" \
  "${APPDIR}/usr/share/icons/hicolor/512x512/apps"

# Bundle relocatable standalone CPython 3.12 + Autum + desktop dependencies
PY_TAG="20250317"
PY_VER="3.12.9"
PY_TAR="cpython-${PY_VER}+${PY_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/${PY_TAR}"

echo "Downloading standalone Python ${PY_VER}..."
curl -fsSL "${PY_URL}" -o "${STAGE}/python.tar.gz"
tar -C "${APPDIR}/usr" -xzf "${STAGE}/python.tar.gz" --strip-components=1

"${APPDIR}/usr/bin/python3" -m pip install --no-cache-dir --upgrade pip
"${APPDIR}/usr/bin/python3" -m pip install --no-cache-dir "${ROOT}[desktop]"

# Uninstall unused PySide6 split wheels (WebEngine ~117 MB, Addons ~59 MB, Pdf ~3 MB)
"${APPDIR}/usr/bin/python3" -m pip uninstall -y PySide6-WebEngine PySide6-Pdf PySide6-Addons 2>/dev/null || true

# Prune any remaining unused PySide6/Qt modules that Autum never imports
SITE_PKG="${APPDIR}/usr/lib/python3.12/site-packages"
PYSIDE_DIR="${SITE_PKG}/PySide6"
if [[ -d "${PYSIDE_DIR}" ]]; then
  UNUSED_QT=(
    WebEngine WebEngineCore WebEngineQuick WebEngineWidgets
    Pdf PdfWidgets
    3DCore 3DExtras 3DInput 3DLogic 3DQuick 3DQuickExtras 3DQuickInput 3DQuickRender 3DQuickScene2D 3DRender
    Quick3D Quick3DAssetImport Quick3DAssetUtils Quick3DEffects Quick3DHelpers Quick3DParticles Quick3DRuntimeRender Quick3DUtils
    Multimedia MultimediaWidgets SpatialAudio
    Charts Graphs DataVisualization
    Designer UiTools Help Linguist
    Sql Location Positioning Bluetooth Sensors Nfc SerialPort SerialBus
    WebSockets WebChannel WebView HttpServer OpcUa RemoteObjects
    VirtualKeyboard TextToSpeech Test StateMachine Scxml
  )
  for mod in "${UNUSED_QT[@]}"; do
    rm -rf \
      "${PYSIDE_DIR}/Qt${mod}"* \
      "${PYSIDE_DIR}/Qt/lib/libQt6${mod}"* \
      "${PYSIDE_DIR}/Qt/qml/Qt${mod}"* \
      "${PYSIDE_DIR}/Qt/plugins/"*"${mod}"* \
      "${PYSIDE_DIR}/Qt/libexec/QtWebEngineProcess" \
      "${PYSIDE_DIR}/Qt/resources/qtwebengine"* \
      "${PYSIDE_DIR}/Qt/translations/qtwebengine"* 2>/dev/null || true
  done
  rm -rf "${PYSIDE_DIR}/examples" "${PYSIDE_DIR}/include" "${PYSIDE_DIR}/typesystems" "${PYSIDE_DIR}/glue" 2>/dev/null || true
fi
find "${SITE_PKG}" -type d \( -name "tests" -o -name "test" -o -name "__pycache__" \) -prune -exec rm -rf {} + 2>/dev/null || true

# Verify pruned bundle still imports Autum and required PySide6 modules cleanly
QT_QPA_PLATFORM=offscreen "${APPDIR}/usr/bin/python3" -c \
  "import autum, PySide6.QtCore, PySide6.QtGui, PySide6.QtQml, PySide6.QtQuick, PySide6.QtNetwork; print('AppDir smoke OK:', autum.__version__)"

install -m 0644 "${ROOT}/contrib/autum.desktop" "${APPDIR}/autum.desktop"
install -m 0644 "${ROOT}/contrib/autum.desktop" "${APPDIR}/usr/share/applications/autum.desktop"
install -m 0644 "${ROOT}/assets/autum-icon.png" "${APPDIR}/autum.png"
install -m 0644 "${ROOT}/assets/autum-icon.png" "${APPDIR}/.DirIcon"
install -m 0644 "${ROOT}/assets/autum-icon.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/autum.png"
install -m 0644 "${ROOT}/assets/autum-orb.png" "${APPDIR}/usr/share/icons/hicolor/512x512/apps/autum.png"

cat > "${APPDIR}/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONHOME="${HERE}/usr"
export PATH="${HERE}/usr/bin:${PATH}"
export QT_PLUGIN_PATH="${HERE}/usr/lib/python3.12/site-packages/PySide6/Qt/plugins${QT_PLUGIN_PATH:+:${QT_PLUGIN_PATH}}"
export QML2_IMPORT_PATH="${HERE}/usr/lib/python3.12/site-packages/PySide6/Qt/qml${QML2_IMPORT_PATH:+:${QML2_IMPORT_PATH}}"
exec "${HERE}/usr/bin/python3" -m autum "$@"
EOF
chmod 0755 "${APPDIR}/AppRun"

APPIMAGETOOL="${STAGE}/appimagetool-x86_64.AppImage"
if command -v appimagetool >/dev/null 2>&1; then
  TOOL="$(command -v appimagetool)"
else
  echo "Downloading appimagetool..."
  curl -fsSL "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" -o "${APPIMAGETOOL}"
  chmod +x "${APPIMAGETOOL}"
  TOOL="${APPIMAGETOOL}"
fi

OUT_APPIMAGE="${OUT_DIR}/Autum-${VERSION}-x86_64.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "${TOOL}" "${APPDIR}" "${OUT_APPIMAGE}"
chmod +x "${OUT_APPIMAGE}"
echo "Built ${OUT_APPIMAGE}"
