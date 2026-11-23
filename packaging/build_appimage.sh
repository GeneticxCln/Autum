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
