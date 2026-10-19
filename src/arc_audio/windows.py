"""Documented Win32 window styles + RegisterHotKey. No hooks of any kind."""

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter

HOTKEY_ID = 0x4152
WM_HOTKEY = 0x0312


def user32():
    dll = ctypes.WinDLL("user32", use_last_error=True)
    dll.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    dll.RegisterHotKey.restype = wintypes.BOOL
    dll.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    dll.UnregisterHotKey.restype = wintypes.BOOL
    return dll


def apply_overlay_styles(hwnd):
    if sys.platform != "win32":
        return
    api = user32()
    pointer = ctypes.c_ssize_t
    get_style = api.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else api.GetWindowLongW
    set_style = api.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else api.SetWindowLongW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = pointer
    set_style.argtypes = [wintypes.HWND, ctypes.c_int, pointer]
    set_style.restype = pointer
    # WS_EX_LAYERED | TRANSPARENT | NOACTIVATE | TOOLWINDOW
    flags = get_style(hwnd, -20) | 0x80000 | 0x20 | 0x8000000 | 0x80
    ctypes.set_last_error(0)
    set_style(hwnd, -20, flags)
    if ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    api.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
    api.SetWindowPos.restype = wintypes.BOOL
    # HWND_TOPMOST, NOMOVE | NOSIZE | NOACTIVATE | FRAMECHANGED
    if not api.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x1 | 0x2 | 0x10 | 0x20):
        raise ctypes.WinError(ctypes.get_last_error())


class EmergencyHotkey(QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.registered = False
        self.message = "Global shortcut is Windows-only; use Stop or the tray menu."
        if sys.platform == "win32":
            self.api = user32()
            # MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, F10. Thread message, no foreground HWND.
            self.registered = bool(self.api.RegisterHotKey(None, HOTKEY_ID, 0x2 | 0x1 | 0x4000, 0x79))
            self.message = ("Global emergency stop: Ctrl+Alt+F10 (registered)" if self.registered else
                            f"Ctrl+Alt+F10 registration FAILED (Windows error {ctypes.get_last_error()}). "
                            "Shortcut may be in use; use tray → Emergency stop or the Stop button.")

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0

    def close(self):
        if self.registered:
            self.api.UnregisterHotKey(None, HOTKEY_ID)
            self.registered = False
