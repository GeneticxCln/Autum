import asyncio
import os
import shutil
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


def cli() -> None:
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            cgroup = fh.read()
    except OSError:
        cgroup = ""
    has_user_bus = bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")) or os.path.exists(
        f"/run/user/{os.getuid()}/bus"
    )
    if has_user_bus and needs_own_scope(cgroup, shutil.which("systemd-run") is not None):
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
