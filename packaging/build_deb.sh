#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')}"
OUT_DIR="${ROOT}/dist"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${OUT_DIR}"
PKG="${STAGE}/autum_${VERSION}_all"

mkdir -p \
  "${PKG}/DEBIAN" \
  "${PKG}/usr/lib/autum" \
  "${PKG}/usr/bin" \
  "${PKG}/usr/share/applications" \
  "${PKG}/usr/share/icons/hicolor/256x256/apps" \
  "${PKG}/usr/share/icons/hicolor/512x512/apps" \
  "${PKG}/usr/lib/systemd/user"

cp -r "${ROOT}/src/autum" "${PKG}/usr/lib/autum/"
find "${PKG}/usr/lib/autum" -name "__pycache__" -type d -exec rm -rf {} +

cat > "${PKG}/usr/bin/autum" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/usr/lib/autum${PYTHONPATH:+:${PYTHONPATH}}"
exec /usr/bin/python3 -m autum "$@"
EOF
chmod 0755 "${PKG}/usr/bin/autum"

install -m 0644 "${ROOT}/contrib/autum.desktop" "${PKG}/usr/share/applications/autum.desktop"
install -m 0644 "${ROOT}/assets/autum-icon.png" "${PKG}/usr/share/icons/hicolor/256x256/apps/autum.png"
install -m 0644 "${ROOT}/assets/autum-orb.png" "${PKG}/usr/share/icons/hicolor/512x512/apps/autum.png"
install -m 0644 "${ROOT}/contrib/autum.service" "${PKG}/usr/lib/systemd/user/autum.service"

INSTALLED_KB="$(du -sk "${PKG}/usr" | awk '{print $1}')"

cat > "${PKG}/DEBIAN/control" <<EOF
Package: autum
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: all
Installed-Size: ${INSTALLED_KB}
Maintainer: GeneticxCln <https://github.com/GeneticxCln/Autum>
Homepage: https://github.com/GeneticxCln/Autum
Depends: python3 (>= 3.12), pipewire-bin | pipewire
Recommends: python3-httpx, python3-numpy, python3-gi
Description: Local, model-agnostic voice-first AI desktop assistant
 Autum provides a floating 3D stardust orb for Linux with push-to-talk,
 wake-word ("Autum"), and hands-free modes powered by local Ollama models
 and PipeWire audio.
EOF

cat > "${PKG}/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
exit 0
EOF
chmod 0755 "${PKG}/DEBIAN/postinst"

DEB_OUT="${OUT_DIR}/autum_${VERSION}_all.deb"
dpkg-deb --root-owner-group --build "${PKG}" "${DEB_OUT}"
echo "Built ${DEB_OUT}"
