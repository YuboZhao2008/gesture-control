"""Windows synthetic mouse input for the opt-in desktop cursor mode."""

import ctypes
import math
import sys

_INPUT_MOUSE = 0
_MOVE = 0x0001
_LEFT_DOWN = 0x0002
_LEFT_UP = 0x0004
_ABSOLUTE = 0x8000
_ABSOLUTE_MAX = 65535
_ESCAPE = 0x1B
_KEY_DOWN = 0x8000
_KEY_PRESSED = 0x0001


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _InputPayload(ctypes.Union):
    # MOUSEINPUT is the largest member of the native INPUT union.
    _fields_ = (("mouse", _MouseInput),)


class _Input(ctypes.Structure):
    _fields_ = (("type", ctypes.c_uint32), ("payload", _InputPayload))


def _load_user32() -> ctypes.CDLL:
    if sys.platform != "win32":
        raise OSError("Desktop cursor mode requires Windows.")
    user32 = ctypes.WinDLL("user32")
    user32.SendInput.argtypes = (ctypes.c_uint32, ctypes.POINTER(_Input), ctypes.c_int)
    user32.SendInput.restype = ctypes.c_uint32
    user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user32.GetAsyncKeyState.restype = ctypes.c_short
    return user32


class WindowsMouse:
    """Drive the primary monitor and track the left button pressed by this backend."""

    def __init__(self) -> None:
        self._user32 = _load_user32()
        self._pressed = False

    def update(self, x: float, y: float, pressed: bool) -> None:
        """Move to normalized screen coordinates and apply a button transition if needed."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Desktop cursor coordinates must be finite.")
        dx = round(min(1.0, max(0.0, x)) * _ABSOLUTE_MAX)
        dy = round(min(1.0, max(0.0, y)) * _ABSOLUTE_MAX)

        flags = _MOVE | _ABSOLUTE
        starting_press = pressed and not self._pressed
        if starting_press:
            flags |= _LEFT_DOWN
            # Cleanup must attempt a release if interrupted during input injection.
            self._pressed = True
        elif self._pressed and not pressed:
            flags |= _LEFT_UP

        if not self._send(flags, dx, dy):
            if starting_press:
                self._pressed = False
            raise OSError("Windows SendInput rejected the mouse event.")
        self._pressed = pressed

    def release(self) -> None:
        """Release a button held by this backend, leaving the cursor in place."""
        if self._pressed:
            if not self._send(_LEFT_UP):
                raise OSError("Windows SendInput rejected the mouse button release.")
            self._pressed = False

    def emergency_stop_requested(self) -> bool:
        """Check Escape even when another app has focus on the active desktop."""
        state = self._user32.GetAsyncKeyState(_ESCAPE)
        # The pressed-since-last-call bit can catch taps between polls on a best-effort basis.
        return bool(state & (_KEY_DOWN | _KEY_PRESSED))

    def _send(self, flags: int, dx: int = 0, dy: int = 0) -> bool:
        mouse = _MouseInput(dx, dy, 0, flags, 0, 0)
        packet = _Input(_INPUT_MOUSE, _InputPayload(mouse=mouse))
        return int(self._user32.SendInput(1, ctypes.pointer(packet), ctypes.sizeof(packet))) == 1
