"""Autum starts in its own scope, so the desktop registers the talk key for Autum."""

from autum.__main__ import needs_own_scope

CLAUDE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-com.anthropic.Claude-163814.scope\n"
OWN_SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-autum-4242.scope\n"
SERVICE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/autum.service\n"


def test_started_from_another_apps_terminal_moves_to_its_own_scope():
    assert needs_own_scope(CLAUDE, have_systemd_run=True)


def test_already_in_its_own_scope_or_service_stays():
    assert not needs_own_scope(OWN_SCOPE, have_systemd_run=True)
    assert not needs_own_scope(SERVICE, have_systemd_run=True)


def test_without_systemd_run_it_just_runs():
    assert not needs_own_scope(CLAUDE, have_systemd_run=False)
