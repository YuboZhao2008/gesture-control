"""Click-through desktop feedback for an active gesture hold."""

import ctypes
import math
import sys

_SIZE = 45
_CENTER = _SIZE // 2

_WS_POPUP = 0x80000000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_HWND_TOPMOST = -1
_SWP_NOSIZE = 0x0001
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_SW_HIDE = 0
_ULW_ALPHA = 0x00000002
_AC_SRC_ALPHA = 1
_INVALID_OBJECT = ctypes.c_void_p(-1).value


class _Point(ctypes.Structure):
    _fields_ = (("x", ctypes.c_int32), ("y", ctypes.c_int32))


class _Size(ctypes.Structure):
    _fields_ = (("cx", ctypes.c_int32), ("cy", ctypes.c_int32))


class _Blend(ctypes.Structure):
    _fields_ = (
        ("operation", ctypes.c_ubyte),
        ("flags", ctypes.c_ubyte),
        ("opacity", ctypes.c_ubyte),
        ("alpha_format", ctypes.c_ubyte),
    )


class _BitmapInfo(ctypes.Structure):
    _fields_ = (
        ("header_size", ctypes.c_uint32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("planes", ctypes.c_uint16),
        ("bits_per_pixel", ctypes.c_uint16),
        ("compression", ctypes.c_uint32),
        ("image_size", ctypes.c_uint32),
        ("x_pixels_per_meter", ctypes.c_int32),
        ("y_pixels_per_meter", ctypes.c_int32),
        ("colors_used", ctypes.c_uint32),
        ("colors_important", ctypes.c_uint32),
        ("colors", ctypes.c_uint32),
    )


def _load_apis() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    if sys.platform != "win32":
        raise OSError("The desktop hold indicator requires Windows.")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    handle = ctypes.c_void_p
    integer = ctypes.c_int
    uint = ctypes.c_uint32
    text = ctypes.c_wchar_p
    point = ctypes.POINTER(_Point)
    size = ctypes.POINTER(_Size)

    user32.CreateWindowExW.argtypes = (
        uint,
        text,
        text,
        uint,
        integer,
        integer,
        integer,
        integer,
        handle,
        handle,
        handle,
        handle,
    )
    user32.CreateWindowExW.restype = handle
    user32.UpdateLayeredWindow.argtypes = (
        handle,
        handle,
        point,
        size,
        handle,
        point,
        uint,
        ctypes.POINTER(_Blend),
        uint,
    )
    user32.UpdateLayeredWindow.restype = integer
    user32.GetCursorPos.argtypes = (point,)
    user32.GetCursorPos.restype = integer
    user32.SetWindowPos.argtypes = (handle, handle, integer, integer, integer, integer, uint)
    user32.SetWindowPos.restype = integer
    user32.ShowWindow.argtypes = (handle, integer)
    user32.ShowWindow.restype = integer
    user32.DestroyWindow.argtypes = (handle,)
    user32.DestroyWindow.restype = integer

    gdi32.CreateCompatibleDC.argtypes = (handle,)
    gdi32.CreateCompatibleDC.restype = handle
    gdi32.CreateDIBSection.argtypes = (
        handle,
        ctypes.POINTER(_BitmapInfo),
        uint,
        ctypes.POINTER(handle),
        handle,
        uint,
    )
    gdi32.CreateDIBSection.restype = handle
    gdi32.SelectObject.argtypes = (handle, handle)
    gdi32.SelectObject.restype = handle
    gdi32.DeleteObject.argtypes = (handle,)
    gdi32.DeleteObject.restype = integer
    gdi32.DeleteDC.argtypes = (handle,)
    gdi32.DeleteDC.restype = integer
    return user32, gdi32


def _ring_pixels() -> bytes:
    # Layered windows take premultiplied BGRA. A dark outer edge and light
    # inner edge keep the ring distinct on both light and dark backgrounds.
    pixels = bytearray()
    for y in range(_SIZE):
        for x in range(_SIZE):
            radius = math.hypot(x - _CENTER, y - _CENTER)
            opacity = max(0.0, min(1.0, radius - 12.5, 21.5 - radius))
            if radius < 15:
                color = (255, 255, 255)
            elif radius < 19:
                color = (184, 239, 34)
            else:
                color = (23, 19, 13)
            pixels.extend(round(component * opacity) for component in (*color, 255))
    return bytes(pixels)


def _install_ring(window: int, user32: ctypes.CDLL, gdi32: ctypes.CDLL) -> None:
    bitmap_info = _BitmapInfo(40, _SIZE, -_SIZE, 1, 32)
    bitmap_bits = ctypes.c_void_p()
    dc = gdi32.CreateCompatibleDC(None)
    if not dc:
        raise OSError("Could not create the desktop hold indicator drawing context.")

    bitmap = None
    original = None
    try:
        bitmap = gdi32.CreateDIBSection(dc, ctypes.byref(bitmap_info), 0, bitmap_bits, None, 0)
        if not bitmap or not bitmap_bits:
            raise OSError("Could not create the desktop hold indicator bitmap.")
        pixels = _ring_pixels()
        ctypes.memmove(bitmap_bits, pixels, len(pixels))
        original = gdi32.SelectObject(dc, bitmap)
        if not original or original == _INVALID_OBJECT:
            raise OSError("Could not select the desktop hold indicator bitmap.")

        size = _Size(_SIZE, _SIZE)
        origin = _Point()
        blend = _Blend(0, 0, 255, _AC_SRC_ALPHA)
        if not user32.UpdateLayeredWindow(
            window, None, None, size, dc, origin, 0, blend, _ULW_ALPHA
        ):
            raise OSError("Could not draw the desktop hold indicator.")
    finally:
        if original and original != _INVALID_OBJECT:
            gdi32.SelectObject(dc, original)
        gdi32.DeleteDC(dc)
        if bitmap:
            gdi32.DeleteObject(bitmap)


class HoldIndicator:
    """Show a ring around the current cursor without activating its window.

    Construct, update, and close the window on the same thread.
    """

    def __init__(self) -> None:
        self._user32, gdi32 = _load_apis()
        self._position: tuple[int, int] | None = None
        styles = (
            _WS_EX_TOPMOST
            | _WS_EX_TRANSPARENT
            | _WS_EX_TOOLWINDOW
            | _WS_EX_LAYERED
            | _WS_EX_NOACTIVATE
        )
        # A stock window class suffices because Windows retains the layered bitmap.
        window = self._user32.CreateWindowExW(
            styles,
            "Static",
            "Gesture cursor hold indicator",
            _WS_POPUP,
            0,
            0,
            _SIZE,
            _SIZE,
            None,
            None,
            None,
            None,
        )
        if not window:
            raise OSError("Could not create the desktop hold indicator window.")
        self._window: int | None = window
        try:
            _install_ring(window, self._user32, gdi32)
        except BaseException:
            self.close()
            raise

    def show(self, visible: bool) -> None:
        """Update the ring at the cursor, or hide it when the hold ends."""
        if self._window is None:
            return
        if not visible:
            if self._position is not None:
                self._user32.ShowWindow(self._window, _SW_HIDE)
                self._position = None
            return

        cursor = _Point()
        if not self._user32.GetCursorPos(ctypes.byref(cursor)):
            raise OSError("Could not read the cursor position for the hold indicator.")
        position = (cursor.x - _CENTER, cursor.y - _CENTER)
        if position != self._position:
            flags = _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW
            if not self._user32.SetWindowPos(self._window, _HWND_TOPMOST, *position, 0, 0, flags):
                raise OSError("Could not position the desktop hold indicator.")
            self._position = position

    def close(self) -> None:
        """Destroy the indicator window."""
        if self._window is not None:
            if not self._user32.DestroyWindow(self._window):
                raise OSError("Could not close the desktop hold indicator.")
            self._window = None
            self._position = None
