#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')}"
OUT_DIR="${ROOT}/dist"
TOPDIR="$(mktemp -d)"
trap 'rm -rf "${TOPDIR}"' EXIT

mkdir -p "${OUT_DIR}" "${TOPDIR}"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

SRC_STAGE="${TOPDIR}/autum-${VERSION}"
mkdir -p "${SRC_STAGE}"
cp -r "${ROOT}/src" "${ROOT}/contrib" "${ROOT}/assets" "${ROOT}/pyproject.toml" "${ROOT}/README.md" "${SRC_STAGE}/"
find "${SRC_STAGE}" -name "__pycache__" -type d -exec rm -rf {} +
tar -C "${TOPDIR}" -czf "${TOPDIR}/SOURCES/autum-${VERSION}.tar.gz" "autum-${VERSION}"

cat > "${TOPDIR}/SPECS/autum.spec" <<EOF
%global debug_package %{nil}
%global __os_install_post %{nil}

Name:           autum
Version:        ${VERSION}
Release:        1%{?dist}
Summary:        Local, model-agnostic voice-first AI desktop assistant
License:        MIT
URL:            https://github.com/GeneticxCln/Autum
Source0:        %{name}-%{version}.tar.gz
BuildArch:      noarch
Requires:       python3 >= 3.12

%description
Autum provides a floating 3D stardust orb for Linux with push-to-talk,
wake-word ("Autum"), and hands-free modes powered by local Ollama models
and PipeWire audio.

%prep
%autosetup

%build

%install
mkdir -p %{buildroot}/usr/lib/autum
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/applications
mkdir -p %{buildroot}/usr/share/icons/hicolor/256x256/apps
mkdir -p %{buildroot}/usr/share/icons/hicolor/512x512/apps
mkdir -p %{buildroot}/usr/lib/systemd/user

cp -r src/autum %{buildroot}/usr/lib/autum/
cat > %{buildroot}/usr/bin/autum <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/usr/lib/autum\${PYTHONPATH:+:\${PYTHONPATH}}"
exec /usr/bin/python3 -m autum "\$@"
WRAPPER
chmod 0755 %{buildroot}/usr/bin/autum

install -m 0644 contrib/autum.desktop %{buildroot}/usr/share/applications/autum.desktop
install -m 0644 assets/autum-icon.png %{buildroot}/usr/share/icons/hicolor/256x256/apps/autum.png
install -m 0644 assets/autum-orb.png %{buildroot}/usr/share/icons/hicolor/512x512/apps/autum.png
install -m 0644 contrib/autum.service %{buildroot}/usr/lib/systemd/user/autum.service

%files
/usr/bin/autum
/usr/lib/autum
/usr/share/applications/autum.desktop
/usr/share/icons/hicolor/256x256/apps/autum.png
/usr/share/icons/hicolor/512x512/apps/autum.png
/usr/lib/systemd/user/autum.service

%post
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
EOF

rpmbuild --define "_topdir ${TOPDIR}" -bb "${TOPDIR}/SPECS/autum.spec"
cp "${TOPDIR}"/RPMS/noarch/autum-*.rpm "${OUT_DIR}/"
echo "Built $(ls "${OUT_DIR}"/autum-*.rpm)"
