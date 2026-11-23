import asyncio
import os
import shutil
import subprocess
import sys

from autum.app import main


def needs_own_scope(cgroup: str, have_systemd_run: bool) -> bool:
    """True when Autum runs inside another app's scope (e.g. a terminal in Claude or an IDE).

    The desktop identifies an app by its systemd scope, so the hold-to-talk
    key would be registered for that other app instead of Autum.
    """
    if not have_systemd_run:
        return False
    return "/app-autum-" not in cgroup and "/autum.service" not in cgroup


def _systemd_user_available() -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    try:
        res = subprocess.run(
            [systemctl, "--user", "show-environment"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
            check=False,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def cli() -> None:
    if os.environ.get("AUTUM_SCOPE_ATTEMPTED") != "1":
        try:
            with open("/proc/self/cgroup", encoding="utf-8") as fh:
                cgroup = fh.read()
        except OSError:
            cgroup = ""
        has_user_bus = bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")) or os.path.exists(
            f"/run/user/{os.getuid()}/bus"
        )
        if (
            has_user_bus
            and needs_own_scope(cgroup, shutil.which("systemd-run") is not None)
            and _systemd_user_available()
        ):
            os.environ["AUTUM_SCOPE_ATTEMPTED"] = "1"
            try:
                os.execvp("systemd-run", [
                    "systemd-run", "--user", "--scope", "--quiet", f"--unit=app-autum-{os.getpid()}",
                    "--", sys.executable, "-m", "autum", *sys.argv[1:],
                ])
            except OSError:
                pass  # run here instead; only the key's registration is affected
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli()
