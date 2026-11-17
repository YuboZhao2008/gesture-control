"""Exercise Windows cursor state transitions without injecting actual input."""

import ctypes
from unittest.mock import Mock

import pytest

from gesture_control import windows_mouse
from gesture_control.windows_mouse import WindowsMouse


@pytest.fixture
def user32(monkeypatch: pytest.MonkeyPatch) -> Mock:
    user32 = Mock()
    user32.SendInput.return_value = 1
    user32.GetAsyncKeyState.return_value = 0
    monkeypatch.setattr(windows_mouse, "_load_user32", lambda: user32)
    return user32


def packets(user32: Mock) -> list[tuple[int, int, int]]:
    result = []
    for call in user32.SendInput.call_args_list:
        count, pointer, size = call.args
        assert count == 1
        assert pointer.contents.type == 0
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        mouse = pointer.contents.payload.mouse
        result.append((mouse.dwFlags, mouse.dx, mouse.dy))
    return result


def test_windows_mouse_maps_primary_monitor_and_emits_one_button_transition(
    user32: Mock,
) -> None:
    mouse = WindowsMouse()
    mouse.release()
    mouse.update(-0.1, 1.1, False)
    mouse.update(0.5, 0.5, True)
    mouse.update(1, 0, True)
    mouse.update(1, 0, False)
    mouse.release()
    assert packets(user32) == [
        (0x8001, 0, 65535),
        (0x8003, 32768, 32768),
        (0x8001, 65535, 0),
        (0x8005, 65535, 0),
    ]


def test_windows_mouse_release_keeps_position_and_is_idempotent(user32: Mock) -> None:
    mouse = WindowsMouse()
    mouse.update(0.5, 0.5, True)
    mouse.release()
    mouse.release()

    assert packets(user32) == [(0x8003, 32768, 32768), (0x0004, 0, 0)]


def test_windows_mouse_retries_failed_release(user32: Mock) -> None:
    mouse = WindowsMouse()
    user32.SendInput.side_effect = [1, 0, 1]
    mouse.update(0.5, 0.5, True)
    with pytest.raises(OSError, match="rejected"):
        mouse.update(1, 0, False)
    mouse.release()

    assert packets(user32) == [(0x8003, 32768, 32768), (0x8005, 65535, 0), (0x0004, 0, 0)]


def test_windows_mouse_does_not_release_a_rejected_press(user32: Mock) -> None:
    mouse = WindowsMouse()
    user32.SendInput.return_value = 0
    with pytest.raises(OSError, match="rejected"):
        mouse.update(0, 1, True)
    mouse.release()

    assert packets(user32) == [(0x8003, 0, 65535)]


def test_windows_mouse_releases_after_interrupted_press(user32: Mock) -> None:
    mouse = WindowsMouse()
    user32.SendInput.side_effect = [KeyboardInterrupt, 1]
    with pytest.raises(KeyboardInterrupt):
        mouse.update(0, 1, True)
    mouse.release()

    assert packets(user32) == [(0x8003, 0, 65535), (0x0004, 0, 0)]


@pytest.mark.parametrize(("state", "requested"), [(0, False), (-32768, True), (1, True)])
def test_windows_mouse_emergency_stop_polls_escape(
    user32: Mock, state: int, requested: bool
) -> None:
    user32.GetAsyncKeyState.return_value = state
    assert WindowsMouse().emergency_stop_requested() is requested
    user32.GetAsyncKeyState.assert_called_once_with(0x1B)
