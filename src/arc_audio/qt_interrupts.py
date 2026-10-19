"""Service console interrupts without raising KeyboardInterrupt inside Qt slots."""

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog


class ConsoleInterrupts:
    """Keep Python signal handling responsive while Qt owns the main event loop.

    The Python handler only records a flag; the timer invokes normal Qt shutdown.
    SIGBREAK is the Windows Ctrl+Break fallback. Previous handlers are restored.
    """

    def __init__(self, request_close):
        self.request_close = request_close
        self.pending = False
        self.interrupted = False
        self.previous = {}
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)

    def receive(self, _number, _frame):
        self.pending = True
        self.interrupted = True

    def poll(self):
        if self.pending:
            self.pending = False
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QDialog):
                modal.reject()
            self.request_close()

    def __enter__(self):
        try:
            for name in ("SIGINT", "SIGBREAK"):
                number = getattr(signal, name, None)
                if number is not None:
                    self.previous[number] = signal.signal(number, self.receive)
            if sys.platform == "win32":
                # Windows child processes can inherit an ignore-Ctrl+C flag from
                # their launcher. Installing a Python SIGINT handler alone does
                # not clear that console attribute.
                import ctypes
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                enable = kernel.SetConsoleCtrlHandler
                enable.argtypes = [ctypes.c_void_p, wintypes.BOOL]
                enable.restype = wintypes.BOOL
                if not enable(None, False) and ctypes.get_last_error() != 6:
                    raise ctypes.WinError(ctypes.get_last_error())
            self.timer.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_exception):
        self.timer.stop()
        for number, previous in self.previous.items():
            signal.signal(number, previous if previous is not None else signal.SIG_DFL)
        self.previous.clear()


def run_event_loop(app, request_close):
    with ConsoleInterrupts(request_close) as interrupts:
        result = app.exec()
    return 130 if interrupts.interrupted else result
